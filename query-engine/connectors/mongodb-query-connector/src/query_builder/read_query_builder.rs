use super::group_by_builder::*;

use crate::{
    BsonTransform, IntoBson,
    constants::*,
    cursor::{CursorBuilder, CursorData},
    filter::{FilterPrefix, MongoFilterVisitor},
    join::JoinStage,
    orderby::OrderByBuilder,
    query_strings::Aggregate,
    root_queries::observing,
    vacuum_cursor,
};
use bson::{Bson, Document, doc};
use itertools::Itertools;
use mongodb::{ClientSession, Collection, options::AggregateOptions};
use query_structure::{
    self as qs, AggregationSelection, CompositeCondition, ConditionListValue, ConditionValue, FieldSelection, Filter,
    Model, PrismaValue, QueryArguments, QueryMode, ScalarCondition, ScalarFieldRef, ScalarListCondition,
    ScalarProjection, Take, VirtualSelection,
};
use std::collections::HashSet;
use std::convert::TryFrom;
use std::future::IntoFuture;

// Mongo Driver broke usage of the simple API, can't be used by us anymore.
// As such the read query will always be based on aggregation pipeline
// such pipeline will have different stages. See
// https://www.mongodb.com/docs/manual/core/aggregation-pipeline/
pub struct ReadQuery {
    pub(crate) stages: Vec<Document>,
}

impl ReadQuery {
    pub async fn execute(
        self,
        on_collection: Collection<Document>,
        with_session: &mut ClientSession,
    ) -> crate::Result<Vec<Document>> {
        let opts = AggregateOptions::builder().allow_disk_use(true).build();
        let query_string_builder = Aggregate::new(&self.stages, on_collection.name());
        let cursor = observing(&query_string_builder, || {
            on_collection
                .aggregate(self.stages.clone())
                .with_options(opts)
                .session(&mut *with_session)
                .into_future()
        })
        .await?;

        vacuum_cursor(cursor, with_session).await
    }
}

/// Translated query arguments ready to use in mongo find or aggregation queries.
#[derive(Debug)]
pub(crate) struct MongoReadQueryBuilder {
    pub(crate) model: Model,

    /// Pre-join, "normal" filters (aggregation expression syntax, wrapped in $expr).
    pub(crate) query: Option<Document>,

    /// Native MongoDB query filter (standard MQL syntax, used directly in $match without $expr).
    /// Takes precedence over `query` when set. Produced for simple scalar equality filters
    /// that can bypass the aggregation expression path, allowing MongoDB to use indexes.
    pub(crate) native_query: Option<Document>,

    /// Join stages.
    pub(crate) joins: Vec<JoinStage>,

    /// Filters that can only be applied after the joins
    /// or aggregations added the required data to execute them.
    pub(crate) join_filters: Vec<Document>,

    /// Aggregation-related stages.
    pub(crate) aggregations: Vec<Document>,

    /// Filters that can only be applied after the aggregations
    /// transformed the documents.
    pub(crate) aggregation_filters: Vec<Document>,

    /// Order by builder for deferred processing.
    order_builder: Option<OrderByBuilder>,

    /// Finalized ordering: Order document.
    pub(crate) order: Option<Document>,

    /// Finalized ordering: Necessary Joins
    /// Kept separate as cursor building needs to consider them seperately.
    pub(crate) order_joins: Vec<JoinStage>,

    /// Finalized ordering aggregation computed from the joins
    pub(crate) order_aggregate_projections: Vec<Document>,

    /// Cursor builder for deferred processing.
    cursor_builder: Option<CursorBuilder>,

    /// Struct containing data required to build cursor queries.
    pub(crate) cursor_data: Option<CursorData>,

    /// Skip a number of documents at the start of the result.
    pub(crate) skip: Option<u64>,

    /// Take only a certain number of documents from the result.
    pub(crate) limit: Option<i64>,

    /// Projection document to scope down return fields.
    pub(crate) projection: Option<Document>,

    /// Switch to indicate the underlying aggregation is a `group_by` query.
    /// This is due to legacy drift in how `aggregate` and `group_by` work in
    /// the API and will hopefully be merged again in the future.
    pub(crate) is_group_by_query: bool,
}

impl MongoReadQueryBuilder {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            query: None,
            native_query: None,
            joins: vec![],
            join_filters: vec![],
            aggregations: vec![],
            aggregation_filters: vec![],
            order_builder: None,
            order: None,
            order_joins: vec![],
            order_aggregate_projections: vec![],
            cursor_builder: None,
            cursor_data: None,
            skip: None,
            limit: None,
            projection: None,
            is_group_by_query: false,
        }
    }

    pub(crate) fn from_args(args: QueryArguments) -> crate::Result<MongoReadQueryBuilder> {
        let reverse_order = args.take.is_reversed();
        let order_by = args.order_by;

        let order_builder = Some(OrderByBuilder::new(order_by.clone(), reverse_order));
        let cursor_builder = args.cursor.map(|c| CursorBuilder::new(c, order_by, reverse_order));

        let mut post_filters = vec![];
        let mut joins = vec![];

        let (query, native_query) = match args.filter {
            Some(filter) => {
                // Try to produce native MQL for simple scalar filters (enables index usage).
                // Falls back to the aggregation expression visitor for complex filters.
                match try_to_native_filter(&filter) {
                    Some(native) => (None, Some(native)),
                    None => {
                        // If a filter comes with joins, it needs to be run _after_ the initial filter query / $matches.
                        let (filter, filter_joins) = MongoFilterVisitor::new(FilterPrefix::default(), false)
                            .visit(filter)?
                            .render();
                        if !filter_joins.is_empty() {
                            joins.extend(filter_joins);
                            post_filters.push(filter);

                            (None, None)
                        } else {
                            (Some(filter), None)
                        }
                    }
                }
            }
            None => (None, None),
        };

        Ok(MongoReadQueryBuilder {
            model: args.model,
            query,
            native_query,
            join_filters: post_filters,
            joins,
            order_builder,
            cursor_builder,
            skip: skip(args.skip.map(|i| i as u64), args.ignore_skip),
            limit: take(args.take, args.ignore_take),
            aggregations: vec![],
            aggregation_filters: vec![],
            order: None,
            order_joins: vec![],
            order_aggregate_projections: vec![],
            cursor_data: None,
            projection: None,
            is_group_by_query: false,
        })
    }

    /// Finalizes the builder and builds a `MongoQuery`.
    pub(crate) fn build(mut self) -> crate::Result<ReadQuery> {
        self.finalize()?;
        Ok(self.build_pipeline_query())
    }

    /// Aggregation-pipeline based query. A distinction must be made between cursored and uncursored queries,
    /// as they require different stage shapes (see individual fns for details).
    fn build_pipeline_query(self) -> ReadQuery {
        let stages = if self.cursor_data.is_none() {
            self.into_pipeline_stages()
        } else {
            self.cursored_pipeline_stages()
        };

        ReadQuery { stages }
    }

    fn into_pipeline_stages(self) -> Vec<Document> {
        let mut stages = vec![];

        // Initial $matches — native MQL (index-friendly) takes precedence over $expr
        if let Some(native) = self.native_query {
            stages.push(doc! { "$match": native })
        } else if let Some(query) = self.query {
            stages.push(doc! { "$match": { "$expr": query } })
        };

        // Joins ($lookup)
        let joins = self.joins.into_iter().chain(self.order_joins);

        let mut unwinds: Vec<Document> = vec![];

        for join_stage in joins {
            let (join, unwind) = join_stage.build();

            if let Some(u) = unwind {
                unwinds.push(u);
            }

            stages.push(join);
        }

        // Order by aggregate computed from joins ($addFields)
        stages.extend(self.order_aggregate_projections);

        // Post-join $matches
        stages.extend(
            self.join_filters
                .into_iter()
                .map(|filter| doc! { "$match": { "$expr": filter } }),
        );

        // If the query is a group by, then skip, take, sort all apply to the _groups_, not the documents
        // before. If it is a plain aggregation, then the aggregate stages need to be _after_ these, because
        // they apply to the documents to be aggregated, not the aggregations (legacy meh).
        if self.is_group_by_query {
            // Aggregates
            stages.extend(self.aggregations.clone());

            // Aggregation filters
            stages.extend(
                self.aggregation_filters
                    .clone()
                    .into_iter()
                    .map(|filter| doc! { "$match": { "$expr": filter } }),
            );
        }

        // Join's $unwind placed before sorting
        // because Mongo does not support sorting multiple arrays
        // https://jira.mongodb.org/browse/SERVER-32859
        stages.extend(unwinds);

        // $sort
        if let Some(order) = self.order {
            stages.push(doc! { "$sort": order })
        };

        // $skip
        if let Some(skip) = self.skip {
            stages.push(doc! { "$skip": i64::try_from(skip).unwrap() });
        };

        // $limit
        if let Some(limit) = self.limit {
            stages.push(doc! { "$limit": limit });
        };

        // $project
        if let Some(projection) = self.projection {
            stages.push(doc! { "$project": projection });
        };

        if !self.is_group_by_query {
            // Aggregates
            stages.extend(self.aggregations);

            // Aggregation filters
            stages.extend(
                self.aggregation_filters
                    .into_iter()
                    .map(|filter| doc! { "$match": { "$expr": filter } }),
            );
        }

        stages
    }

    /// Pipeline query with a cursor. Requires special building to form a query that first
    /// pins a cursor and then builds cursor conditions based on that cursor document
    /// and the orderings that the query defined.
    /// The stages have the form:
    /// ```text
    /// testModel.aggregate([
    ///     { $match: { <filter finding exactly one document (cursor doc)> }},
    ///     { $lookup: { <if present, join that are required for orderBy relations> }}
    ///     { ... more joins if necessary ... }
    ///     {
    ///         $lookup: <"self join" testModel and execute non-cursor query with cursor filter here.>
    ///     }
    /// ])
    /// ```
    /// Expressed in words, this query first makes the cursor document (that defines all values
    /// to make cursor comparators work) available for the inner pipeline to build the filters.
    /// The inner pipeline is basically what an non-cursor query would look like with added cursor
    /// conditions. The inner join stage is refered to as a self-join here because it joins the cursor document
    /// to it's collection again to pull in all documents for filtering, but technically it doesn't
    /// actually join anything.
    ///
    /// Todo concrete example
    fn cursored_pipeline_stages(mut self) -> Vec<Document> {
        let coll_name = self.model.db_name().to_owned();
        let cursor_data = self.cursor_data.take().unwrap();

        // For now we assume that simply putting the cursor condition into the join conditions is enough
        // to let them run in the correct place.
        self.join_filters.push(cursor_data.cursor_condition);

        let order_join_stages = self
            .order_joins
            .clone()
            .into_iter()
            .map(|nested_stage| {
                let (join, _) = nested_stage.build();

                join
            })
            .collect_vec();

        // Outer query to pin the cursor document.
        let mut outer_stages = vec![];

        // First match the cursor, then add required ordering joins.
        outer_stages.push(doc! { "$match": { "$expr": cursor_data.cursor_filter } });
        outer_stages.extend(order_join_stages);

        outer_stages.extend(self.order_aggregate_projections.clone());

        // Self-"join" collection
        let inner_stages = self.into_pipeline_stages();

        outer_stages.push(doc! {
            "$lookup": {
                "from": coll_name,
                "let": cursor_data.bindings,
                "pipeline": inner_stages,
                "as": "cursor_inner",
            }
        });

        outer_stages.push(doc! { "$unwind": "$cursor_inner" });
        outer_stages.push(doc! { "$replaceRoot": { "newRoot": "$cursor_inner" } });

        outer_stages
    }

    /// Adds a final projection onto the fields specified by the `FieldSelection`.
    pub fn with_model_projection(mut self, selected_fields: FieldSelection) -> crate::Result<Self> {
        let projection = selected_fields.into_bson()?.into_document()?;
        self.projection = Some(projection);

        Ok(self)
    }

    /// Adds the necessary joins and the associated selections to the projection
    pub fn with_virtual_fields<'a>(
        mut self,
        virtual_selections: impl Iterator<Item = &'a VirtualSelection>,
    ) -> crate::Result<Self> {
        for aggr in virtual_selections {
            let join = match aggr {
                VirtualSelection::RelationCount(rf, filter) => {
                    let filter = filter
                        .as_ref()
                        .map(|f| MongoFilterVisitor::new(FilterPrefix::default(), false).visit(f.clone()))
                        .transpose()?;

                    JoinStage {
                        source: rf.clone(),
                        alias: Some(aggr.db_alias()),
                        nested: vec![],
                        filter,
                    }
                }
            };

            let projection = doc! {
              aggr.db_alias(): { "$size": format!("${}", aggr.db_alias()) }
            };

            self.joins.push(join);
            self.projection = self.projection.map_or(Some(projection.clone()), |mut p| {
                p.extend(projection);
                Some(p)
            });
        }

        Ok(self)
    }

    /// Adds group-by fields with their aggregations to this query.
    pub fn with_groupings(
        mut self,
        by_fields: Vec<ScalarFieldRef>,
        selections: &[AggregationSelection],
        having: Option<Filter>,
    ) -> crate::Result<Self> {
        if !by_fields.is_empty() {
            self.is_group_by_query = true;
        }

        let mut group_by = GroupByBuilder::new();
        group_by.with_selections(selections);

        if let Some(having) = having {
            group_by.with_having_filter(&having);

            // Having filters can only appear in group by queries.
            // All group by fields go into the UNDERSCORE_ID key of the result document.
            // As it is the only place where the flat scalars are contained for the group,
            // we need to refer to that object.
            let prefix = FilterPrefix::from(group_by::UNDERSCORE_ID);
            let (filter_doc, _) = MongoFilterVisitor::new(prefix, false).visit(having)?.render();

            self.aggregation_filters.push(filter_doc);
        }

        let (grouping_stage, project_stage) = group_by.render(by_fields);

        self.aggregations.push(doc! { "$group": grouping_stage });

        if let Some(project_stage) = project_stage {
            self.aggregations.push(doc! { "$project": project_stage });
        }

        Ok(self)
    }

    /// Runs last transformations on `self` to execute steps dependent on base args.
    fn finalize(&mut self) -> crate::Result<()> {
        // Cursor building depends on the ordering, so it must come first.
        if let Some(order_builder) = self.order_builder.take() {
            let (order, order_aggregate_projections, joins) = order_builder.build(self.is_group_by_query);

            self.order_joins.extend(joins);
            self.order = order;
            self.order_aggregate_projections = order_aggregate_projections;
        }

        if let Some(cursor_builder) = self.cursor_builder.take() {
            let cursor_data = cursor_builder.build()?;

            self.cursor_data = Some(cursor_data);
        }

        Ok(())
    }
}

fn skip(skip: Option<u64>, ignore: bool) -> Option<u64> {
    if ignore { None } else { skip }
}

fn take(take: Take, ignore: bool) -> Option<i64> {
    if ignore {
        None
    } else {
        match take {
            Take::All => None,
            Take::One | Take::NegativeOne => Some(1),
            Take::Some(n) => Some(n.abs()),
        }
    }
}

/// Public helper for write paths that want to reuse the same native translation
/// used by read query argument translation.
pub(crate) fn try_filter_to_native_mql(filter: &Filter) -> Option<Document> {
    try_to_native_filter(filter)
}

/// Try to convert a Filter AST directly into native MongoDB query syntax.
///
/// Native MQL filters (`{ field: value }`) can be used in `$match` without `$expr`,
/// allowing MongoDB to use indexes (IXSCAN) instead of full collection scans (COLLSCAN).
///
fn try_to_native_filter(filter: &Filter) -> Option<Document> {
    try_to_native_filter_with_prefix(filter, None)
}

fn try_to_native_filter_with_prefix(filter: &Filter, path_prefix: Option<&str>) -> Option<Document> {
    match filter {
        Filter::Scalar(sf) => try_scalar_to_native(sf, path_prefix),
        Filter::ScalarList(slf) => try_scalar_list_to_native(slf, path_prefix),
        Filter::And(filters) => {
            let mut natives: Vec<Document> = filters
                .iter()
                .map(|f| try_to_native_filter_with_prefix(f, path_prefix))
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .filter(|d| !d.is_empty())
                .collect();

            if natives.is_empty() {
                return Some(native_true_filter());
            }

            if natives.len() == 1 {
                return natives.pop();
            }

            if can_flatten_native_and(&natives) {
                let mut merged = Document::new();
                for native in natives {
                    merged.extend(native);
                }
                Some(merged)
            } else {
                Some(doc! { "$and": natives })
            }
        }
        Filter::Or(filters) => {
            let natives: Vec<Document> = filters
                .iter()
                .map(|f| try_to_native_filter_with_prefix(f, path_prefix))
                .collect::<Option<Vec<_>>>()?;

            if natives.is_empty() {
                // OR([]) is always false.
                return Some(native_false_filter());
            }

            // OR([true, ...]) is always true.
            if natives.iter().any(|d| d.is_empty()) {
                return Some(native_true_filter());
            }

            if natives.len() == 1 {
                return natives.into_iter().next();
            }

            Some(doc! { "$or": natives })
        }
        Filter::Not(filters) => {
            let natives: Vec<Document> = filters
                .iter()
                .map(|f| try_to_native_filter_with_prefix(f, path_prefix))
                .collect::<Option<Vec<_>>>()?;

            if natives.is_empty() {
                return Some(native_true_filter());
            }

            Some(doc! { "$nor": natives })
        }
        Filter::Composite(cf) => {
            let nested_prefix = join_field_path(path_prefix, cf.field.db_name());
            match cf.condition.as_ref() {
                CompositeCondition::Is(inner) => try_to_native_filter_with_prefix(inner, Some(nested_prefix.as_str())),
                CompositeCondition::IsNot(inner) => {
                    let nested = try_to_native_filter_with_prefix(inner, Some(nested_prefix.as_str()))?;
                    Some(doc! { "$nor": [nested] })
                }
                CompositeCondition::IsSet(is_set) => Some(doc! { nested_prefix: { "$exists": *is_set } }),
                _ => None,
            }
        }
        Filter::Empty => Some(Document::new()),
        _ => None,
    }
}

/// Flat merge is only safe when each child document has distinct top-level
/// field keys and no top-level operators (keys starting with `$`).
fn can_flatten_native_and(natives: &[Document]) -> bool {
    let mut seen = HashSet::new();

    for native in natives {
        for key in native.keys() {
            if key.starts_with('$') {
                return false;
            }

            if !seen.insert(key.as_str()) {
                return false;
            }
        }
    }

    true
}

/// Convert a single ScalarFilter to native MQL.
///
/// Handles: equality, comparisons, in/notIn, string ops, isSet.
/// Returns None for: field refs, complex/unsupported compound projections and JSON/search conditions.
fn try_scalar_to_native(sf: &qs::ScalarFilter, path_prefix: Option<&str>) -> Option<Document> {
    let field: &ScalarFieldRef = match &sf.projection {
        ScalarProjection::Single(f) => f,
        ScalarProjection::Compound(_) => return None,
    };

    let name = join_field_path(path_prefix, field.db_name());
    let insensitive = sf.mode == QueryMode::Insensitive;

    // Only support known query modes.
    if !matches!(sf.mode, QueryMode::Default | QueryMode::Insensitive) {
        return None;
    }

    match &sf.condition {
        // === Equality ===
        ScalarCondition::Equals(ConditionValue::Value(pv)) => {
            if matches!(pv, PrismaValue::Null) {
                // Prisma null equality excludes missing fields. Native { field: null }
                // includes missing fields, so keep an explicit $exists guard.
                return Some(doc! { "$and": [
                    { &name: { "$exists": true } },
                    { &name: Bson::Null }
                ] });
            }

            if insensitive {
                let s = pv.as_string()?;
                return Some(doc! {
                    &name: regex_filter_doc(format!("^{}$", regex::escape(s)), true)
                });
            }

            let v = (field, pv.clone()).into_bson().ok()?;
            Some(doc! { &name: v })
        }

        // === NotEquals ===
        // Native $ne includes docs where field is missing, but $expr excludes them.
        // Add $exists guard to match $expr semantics.
        ScalarCondition::NotEquals(ConditionValue::Value(pv)) => {
            if matches!(pv, PrismaValue::Null) {
                // Prisma's `not: null` should exclude both explicit null and missing fields.
                // Native `$ne: null` alone would include missing docs, so keep the `$exists` guard.
                return Some(doc! { "$and": [
                    { &name: { "$exists": true } },
                    { &name: { "$ne": Bson::Null } }
                ] });
            }

            if insensitive {
                let s = pv.as_string()?;
                return Some(doc! { "$and": [
                    { &name: { "$exists": true } },
                    { &name: { "$not": regex_filter_doc(format!("^{}$", regex::escape(s)), true) } }
                ] });
            }

            let v = (field, pv.clone()).into_bson().ok()?;
            Some(doc! { "$and": [
                { &name: { "$exists": true } },
                { &name: { "$ne": v } }
            ] })
        }

        // === Comparison operators ===
        ScalarCondition::LessThan(ConditionValue::Value(pv)) => {
            if insensitive {
                return None;
            }
            let v = (field, pv.clone()).into_bson().ok()?;
            Some(doc! { &name: { "$lt": v } })
        }
        ScalarCondition::LessThanOrEquals(ConditionValue::Value(pv)) => {
            if insensitive {
                return None;
            }
            let v = (field, pv.clone()).into_bson().ok()?;
            Some(doc! { &name: { "$lte": v } })
        }
        ScalarCondition::GreaterThan(ConditionValue::Value(pv)) => {
            if insensitive {
                return None;
            }
            let v = (field, pv.clone()).into_bson().ok()?;
            Some(doc! { &name: { "$gt": v } })
        }
        ScalarCondition::GreaterThanOrEquals(ConditionValue::Value(pv)) => {
            if insensitive {
                return None;
            }
            let v = (field, pv.clone()).into_bson().ok()?;
            Some(doc! { &name: { "$gte": v } })
        }

        // === In ===
        ScalarCondition::In(ConditionListValue::List(vals)) => {
            if insensitive {
                return None;
            }
            // Bail if any null values — native $in with null matches missing docs too
            if vals.iter().any(|v| matches!(v, PrismaValue::Null)) {
                return None;
            }
            let arr: Vec<Bson> = vals
                .iter()
                .map(|v| (field, v.clone()).into_bson())
                .collect::<Result<_, _>>()
                .ok()?;
            Some(doc! { &name: { "$in": arr } })
        }

        // === NotIn ===
        // Same $exists guard as NotEquals — native $nin includes missing docs.
        ScalarCondition::NotIn(ConditionListValue::List(vals)) => {
            if insensitive {
                return None;
            }
            if vals.iter().any(|v| matches!(v, PrismaValue::Null)) {
                return None;
            }
            let arr: Vec<Bson> = vals
                .iter()
                .map(|v| (field, v.clone()).into_bson())
                .collect::<Result<_, _>>()
                .ok()?;
            Some(doc! { "$and": [
                { &name: { "$exists": true } },
                { &name: { "$nin": arr } }
            ] })
        }

        // === String operations ===
        ScalarCondition::StartsWith(ConditionValue::Value(pv)) => {
            let s = pv.as_string()?;
            Some(doc! { &name: regex_filter_doc(format!("^{}", regex::escape(s)), insensitive) })
        }
        ScalarCondition::EndsWith(ConditionValue::Value(pv)) => {
            let s = pv.as_string()?;
            Some(doc! { &name: regex_filter_doc(format!("{}$", regex::escape(s)), insensitive) })
        }
        ScalarCondition::Contains(ConditionValue::Value(pv)) => {
            let s = pv.as_string()?;
            Some(doc! { &name: regex_filter_doc(regex::escape(s), insensitive) })
        }
        ScalarCondition::NotStartsWith(ConditionValue::Value(pv)) => {
            let s = pv.as_string()?;
            Some(doc! { "$and": [
                { &name: { "$exists": true } },
                { &name: { "$not": regex_filter_doc(format!("^{}", regex::escape(s)), insensitive) } }
            ] })
        }
        ScalarCondition::NotEndsWith(ConditionValue::Value(pv)) => {
            let s = pv.as_string()?;
            Some(doc! { "$and": [
                { &name: { "$exists": true } },
                { &name: { "$not": regex_filter_doc(format!("{}$", regex::escape(s)), insensitive) } }
            ] })
        }
        ScalarCondition::NotContains(ConditionValue::Value(pv)) => {
            let s = pv.as_string()?;
            Some(doc! { "$and": [
                { &name: { "$exists": true } },
                { &name: { "$not": regex_filter_doc(regex::escape(s), insensitive) } }
            ] })
        }

        // === IsSet ===
        ScalarCondition::IsSet(is_set) => Some(doc! { &name: { "$exists": *is_set } }),

        // Everything else (field refs, JSON, search) → $expr
        _ => None,
    }
}

/// Convert a ScalarListFilter (array field operations) to native MQL.
///
/// Handles: has (Contains), hasSome (ContainsSome), hasEvery (ContainsEvery), isEmpty.
fn try_scalar_list_to_native(slf: &qs::ScalarListFilter, path_prefix: Option<&str>) -> Option<Document> {
    let field = &slf.field;
    let name = join_field_path(path_prefix, field.db_name());

    match &slf.condition {
        // has: value → { field: value } (MongoDB matches if array contains element)
        ScalarListCondition::Contains(ConditionValue::Value(pv)) => {
            let v = (field, pv.clone()).into_bson().ok()?;
            Some(doc! { &name: v })
        }

        // hasSome: [...] → { field: { $in: [...] } }
        ScalarListCondition::ContainsSome(ConditionListValue::List(vals)) if !vals.is_empty() => {
            let arr: Vec<Bson> = vals
                .iter()
                .map(|v| (field, v.clone()).into_bson())
                .collect::<Result<_, _>>()
                .ok()?;
            Some(doc! { &name: { "$in": arr } })
        }

        // hasEvery: [...] → { field: { $all: [...] } }
        ScalarListCondition::ContainsEvery(ConditionListValue::List(vals)) if !vals.is_empty() => {
            let arr: Vec<Bson> = vals
                .iter()
                .map(|v| (field, v.clone()).into_bson())
                .collect::<Result<_, _>>()
                .ok()?;
            Some(doc! { &name: { "$all": arr } })
        }

        // isEmpty: true → { field: { $size: 0 } }
        ScalarListCondition::IsEmpty(true) => Some(doc! { &name: { "$size": 0_i32 } }),

        // isEmpty: false → { "field.0": { $exists: true } }
        ScalarListCondition::IsEmpty(false) => Some(doc! { format!("{}.0", &name): { "$exists": true } }),

        // FieldRef variants, empty lists → fall back to $expr
        _ => None,
    }
}

fn join_field_path(path_prefix: Option<&str>, field_name: &str) -> String {
    match path_prefix {
        Some(prefix) if !prefix.is_empty() => format!("{prefix}.{field_name}"),
        _ => field_name.to_string(),
    }
}

fn regex_filter_doc(pattern: String, insensitive: bool) -> Document {
    if insensitive {
        doc! { "$regex": pattern, "$options": "i" }
    } else {
        doc! { "$regex": pattern }
    }
}

fn native_true_filter() -> Document {
    Document::new()
}

fn native_false_filter() -> Document {
    doc! { "$nor": [Document::new()] }
}
