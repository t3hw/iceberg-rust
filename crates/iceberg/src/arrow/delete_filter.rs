// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use tokio::sync::Notify;
use tokio::sync::oneshot::Receiver;

use super::caching_delete_file_loader::EqDeleteSet;
use crate::delete_vector::DeleteVector;
use crate::scan::{FileScanTask, FileScanTaskDeleteFile};
use crate::spec::DataContentType;
use crate::{Error, ErrorKind, Result};

#[derive(Debug)]
enum EqDelState {
    Loading(Arc<Notify>),
    Loaded(Arc<EqDeleteSet>),
}

/// State tracking for positional delete files.
/// Unlike equality deletes, positional deletes must be fully loaded before
/// the ArrowReader proceeds because retrieval is synchronous and non-blocking.
#[derive(Debug)]
enum PosDelState {
    /// The file is currently being loaded by a task.
    /// The notifier allows other tasks to wait for completion.
    Loading(Arc<Notify>),
    /// The file has been fully loaded and merged into the delete vector map.
    Loaded,
}

#[derive(Debug, Default)]
struct DeleteFileFilterState {
    delete_vectors: HashMap<String, Arc<Mutex<DeleteVector>>>,
    equality_deletes: HashMap<String, EqDelState>,
    positional_deletes: HashMap<String, PosDelState>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DeleteFilter {
    state: Arc<RwLock<DeleteFileFilterState>>,
}

/// Action to take when trying to start loading a positional delete file
pub(crate) enum PosDelLoadAction {
    /// The file is not loaded, the caller should load it.
    Load,
    /// The file is already loaded, nothing to do.
    AlreadyLoaded,
    /// The file is currently being loaded by another task.
    /// The caller *must* wait for this notifier to ensure data availability
    /// before returning, as subsequent access (get_delete_vector) is synchronous.
    WaitFor(Arc<Notify>),
}

impl DeleteFilter {
    /// Retrieve a delete vector for the data file associated with a given file scan task
    pub(crate) fn get_delete_vector(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Option<Arc<Mutex<DeleteVector>>> {
        self.get_delete_vector_for_path(file_scan_task.data_file_path())
    }

    /// Retrieve a delete vector for a data file
    pub(crate) fn get_delete_vector_for_path(
        &self,
        data_file_path: &str,
    ) -> Option<Arc<Mutex<DeleteVector>>> {
        self.state
            .read()
            .ok()
            .and_then(|st| st.delete_vectors.get(data_file_path).cloned())
    }

    pub(crate) fn try_start_eq_del_load(&self, file_path: &str) -> Option<Arc<Notify>> {
        let mut state = self.state.write().unwrap();

        // Skip if already loaded/loading - another task owns it
        if state.equality_deletes.contains_key(file_path) {
            return None;
        }

        // Mark as loading to prevent duplicate work
        let notifier = Arc::new(Notify::new());
        state
            .equality_deletes
            .insert(file_path.to_string(), EqDelState::Loading(notifier.clone()));

        Some(notifier)
    }

    /// Attempts to mark a positional delete file as "loading".
    ///
    /// Returns an action dictating whether the caller should load the file,
    /// wait for another task to load it, or do nothing.
    pub(crate) fn try_start_pos_del_load(&self, file_path: &str) -> PosDelLoadAction {
        let mut state = self.state.write().unwrap();

        if let Some(state) = state.positional_deletes.get(file_path) {
            match state {
                PosDelState::Loaded => return PosDelLoadAction::AlreadyLoaded,
                PosDelState::Loading(notify) => return PosDelLoadAction::WaitFor(notify.clone()),
            }
        }

        let notifier = Arc::new(Notify::new());
        state
            .positional_deletes
            .insert(file_path.to_string(), PosDelState::Loading(notifier));

        PosDelLoadAction::Load
    }

    /// Marks a positional delete file as successfully loaded and notifies any waiting tasks.
    pub(crate) fn finish_pos_del_load(&self, file_path: &str) {
        let notify = {
            let mut state = self.state.write().unwrap();
            if let Some(PosDelState::Loading(notify)) = state
                .positional_deletes
                .insert(file_path.to_string(), PosDelState::Loaded)
            {
                Some(notify)
            } else {
                None
            }
        };

        if let Some(notify) = notify {
            notify.notify_waiters();
        }
    }

    /// Retrieve the equality delete set for a given eq delete file path.
    /// Waits asynchronously if the set is still being loaded.
    pub(crate) async fn get_equality_delete_set_for_delete_file_path(
        &self,
        file_path: &str,
    ) -> Option<Arc<EqDeleteSet>> {
        let notifier = {
            match self.state.read().unwrap().equality_deletes.get(file_path) {
                None => return None,
                Some(EqDelState::Loading(notifier)) => notifier.clone(),
                Some(EqDelState::Loaded(eq_delete_set)) => {
                    return Some(eq_delete_set.clone());
                }
            }
        };

        notifier.notified().await;

        match self.state.read().unwrap().equality_deletes.get(file_path) {
            Some(EqDelState::Loaded(eq_delete_set)) => Some(eq_delete_set.clone()),
            _ => unreachable!("Cannot be any other state than loaded"),
        }
    }

    /// Builds equality delete sets for the provided task.
    ///
    /// Returns a list of delete sets, one per distinct `equality_ids` group.
    /// Most tables use a single `equality_ids` set, so this typically returns
    /// zero or one element. Multiple elements occur only when different delete
    /// files on the same partition use different equality column sets.
    ///
    /// When only one delete file applies for a group, returns the cached `Arc`
    /// directly — no deep clone of the hash set.
    pub(crate) async fn build_equality_delete_sets(
        &self,
        file_scan_task: &FileScanTask,
    ) -> Result<Vec<Arc<EqDeleteSet>>> {
        // Collect all applicable equality delete sets, reusing cached Arcs.
        // Group by field layout so we only union sets with matching columns.
        let mut groups: HashMap<Vec<(String, i32)>, Vec<Arc<EqDeleteSet>>> = HashMap::new();

        for delete in &file_scan_task.deletes {
            if !is_equality_delete(delete) {
                continue;
            }

            let Some(eq_set) = self
                .get_equality_delete_set_for_delete_file_path(&delete.file_path)
                .await
            else {
                return Err(Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "Missing equality delete set for file '{}'",
                        delete.file_path
                    ),
                ));
            };

            if !eq_set.is_empty() {
                groups
                    .entry(eq_set.fields.clone())
                    .or_default()
                    .push(eq_set);
            }
        }

        // For each group, union all sets into one.
        let mut result = Vec::with_capacity(groups.len());
        for (_fields, sets) in groups {
            match sets.len() {
                0 => {}
                // Single file in group: return the cached Arc directly.
                1 => result.push(sets.into_iter().next().unwrap()),
                // Multiple files with same fields: union into a new set.
                _ => {
                    let mut combined = (*sets[0]).clone();
                    for set in &sets[1..] {
                        combined.union(set);
                    }
                    result.push(Arc::new(combined));
                }
            }
        }

        Ok(result)
    }

    pub(crate) fn upsert_delete_vector(
        &mut self,
        data_file_path: String,
        delete_vector: DeleteVector,
    ) {
        let mut state = self.state.write().unwrap();

        let Some(entry) = state.delete_vectors.get_mut(&data_file_path) else {
            state
                .delete_vectors
                .insert(data_file_path, Arc::new(Mutex::new(delete_vector)));
            return;
        };

        *entry.lock().unwrap() |= delete_vector;
    }

    pub(crate) fn insert_equality_delete(
        &self,
        delete_file_path: &str,
        eq_del: Receiver<Arc<EqDeleteSet>>,
    ) {
        let notify = Arc::new(Notify::new());
        {
            let mut state = self.state.write().unwrap();
            state.equality_deletes.insert(
                delete_file_path.to_string(),
                EqDelState::Loading(notify.clone()),
            );
        }

        let state = self.state.clone();
        let delete_file_path = delete_file_path.to_string();
        crate::runtime::spawn(async move {
            let eq_del = eq_del.await.unwrap();
            {
                let mut state = state.write().unwrap();
                state
                    .equality_deletes
                    .insert(delete_file_path, EqDelState::Loaded(eq_del));
            }
            notify.notify_waiters();
        });
    }
}

pub(crate) fn is_equality_delete(f: &FileScanTaskDeleteFile) -> bool {
    matches!(f.file_type, DataContentType::EqualityDeletes)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::Schema as ArrowSchema;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::caching_delete_file_loader::{
        CachingDeleteFileLoader, EqDeleteKey, EqDeleteSet,
    };
    use crate::io::FileIO;
    use crate::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema, Type};

    type ArrowSchemaRef = Arc<ArrowSchema>;

    const FIELD_ID_POSITIONAL_DELETE_FILE_PATH: u64 = 2147483546;
    const FIELD_ID_POSITIONAL_DELETE_POS: u64 = 2147483545;

    #[tokio::test]
    async fn test_delete_file_filter_load_deletes() {
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path();
        let file_io = FileIO::new_with_fs();

        let delete_file_loader = CachingDeleteFileLoader::new(file_io.clone(), 10);

        let file_scan_tasks = setup(table_location);

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[0].deletes, file_scan_tasks[0].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[0])
            .unwrap();
        assert_eq!(result.lock().unwrap().len(), 12); // pos dels from pos del file 1 and 2

        let delete_filter = delete_file_loader
            .load_deletes(&file_scan_tasks[1].deletes, file_scan_tasks[1].schema_ref())
            .await
            .unwrap()
            .unwrap();

        let result = delete_filter
            .get_delete_vector(&file_scan_tasks[1])
            .unwrap();
        assert_eq!(result.lock().unwrap().len(), 8); // no pos dels for file 3
    }

    pub(crate) fn setup(table_location: &Path) -> Vec<FileScanTask> {
        let data_file_schema = Arc::new(Schema::builder().build().unwrap());
        let positional_delete_schema = create_pos_del_schema();

        let file_path_values = [
            vec![format!("{}/1.parquet", table_location.to_str().unwrap()); 8],
            vec![format!("{}/1.parquet", table_location.to_str().unwrap()); 8],
            vec![format!("{}/2.parquet", table_location.to_str().unwrap()); 8],
        ];
        let pos_values = [
            vec![0i64, 1, 3, 5, 6, 8, 1022, 1023],
            vec![0i64, 1, 3, 5, 20, 21, 22, 23],
            vec![0i64, 1, 3, 5, 6, 8, 1022, 1023],
        ];

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        for n in 1..=3 {
            let file_path_vals = file_path_values.get(n - 1).unwrap();
            let file_path_col = Arc::new(StringArray::from_iter_values(file_path_vals));

            let pos_vals = pos_values.get(n - 1).unwrap();
            let pos_col = Arc::new(Int64Array::from_iter_values(pos_vals.clone()));

            let positional_deletes_to_write =
                RecordBatch::try_new(positional_delete_schema.clone(), vec![
                    file_path_col.clone(),
                    pos_col.clone(),
                ])
                .unwrap();

            let file = File::create(format!(
                "{}/pos-del-{}.parquet",
                table_location.to_str().unwrap(),
                n
            ))
            .unwrap();
            let mut writer = ArrowWriter::try_new(
                file,
                positional_deletes_to_write.schema(),
                Some(props.clone()),
            )
            .unwrap();

            writer
                .write(&positional_deletes_to_write)
                .expect("Writing batch");

            // writer must be closed to write footer
            writer.close().unwrap();
        }

        let pos_del_1 = FileScanTaskDeleteFile {
            file_path: format!("{}/pos-del-1.parquet", table_location.to_str().unwrap()),
            file_size_in_bytes: std::fs::metadata(format!(
                "{}/pos-del-1.parquet",
                table_location.to_str().unwrap()
            ))
            .unwrap()
            .len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
        };

        let pos_del_2 = FileScanTaskDeleteFile {
            file_path: format!("{}/pos-del-2.parquet", table_location.to_str().unwrap()),
            file_size_in_bytes: std::fs::metadata(format!(
                "{}/pos-del-2.parquet",
                table_location.to_str().unwrap()
            ))
            .unwrap()
            .len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
        };

        let pos_del_3 = FileScanTaskDeleteFile {
            file_path: format!("{}/pos-del-3.parquet", table_location.to_str().unwrap()),
            file_size_in_bytes: std::fs::metadata(format!(
                "{}/pos-del-3.parquet",
                table_location.to_str().unwrap()
            ))
            .unwrap()
            .len(),
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
        };

        let file_scan_tasks = vec![
            FileScanTask {
                file_size_in_bytes: 0,
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: format!("{}/1.parquet", table_location.to_str().unwrap()),
                data_file_format: DataFileFormat::Parquet,
                schema: data_file_schema.clone(),
                project_field_ids: vec![],
                predicate: None,
                deletes: vec![pos_del_1, pos_del_2.clone()],
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
            },
            FileScanTask {
                file_size_in_bytes: 0,
                start: 0,
                length: 0,
                record_count: None,
                data_file_path: format!("{}/2.parquet", table_location.to_str().unwrap()),
                data_file_format: DataFileFormat::Parquet,
                schema: data_file_schema.clone(),
                project_field_ids: vec![],
                predicate: None,
                deletes: vec![pos_del_3],
                partition: None,
                partition_spec: None,
                name_mapping: None,
                case_sensitive: false,
            },
        ];

        file_scan_tasks
    }

    pub(crate) fn create_pos_del_schema() -> ArrowSchemaRef {
        let fields = vec![
            arrow_schema::Field::new("file_path", arrow_schema::DataType::Utf8, false)
                .with_metadata(HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    FIELD_ID_POSITIONAL_DELETE_FILE_PATH.to_string(),
                )])),
            arrow_schema::Field::new("pos", arrow_schema::DataType::Int64, false).with_metadata(
                HashMap::from([(
                    PARQUET_FIELD_ID_META_KEY.to_string(),
                    FIELD_ID_POSITIONAL_DELETE_POS.to_string(),
                )]),
            ),
        ];
        Arc::new(arrow_schema::Schema::new(fields))
    }

    #[tokio::test]
    async fn test_build_equality_delete_set_unions_multiple_files() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                ])
                .build()
                .unwrap(),
        );

        let task = FileScanTask {
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: "data.parquet".to_string(),
            data_file_format: crate::spec::DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![
                FileScanTaskDeleteFile {
                    file_path: "eq-del-1.parquet".to_string(),
                    file_size_in_bytes: 1,
                    file_type: DataContentType::EqualityDeletes,
                    partition_spec_id: 0,
                    equality_ids: Some(vec![1]),
                },
                FileScanTaskDeleteFile {
                    file_path: "eq-del-2.parquet".to_string(),
                    file_size_in_bytes: 1,
                    file_type: DataContentType::EqualityDeletes,
                    partition_spec_id: 0,
                    equality_ids: Some(vec![1]),
                },
            ],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let filter = DeleteFilter::default();

        // Insert two equality delete sets with different keys
        let mut set1 = EqDeleteSet {
            keys: std::collections::HashSet::new(),
            fields: vec![("id".to_string(), 1)],
        };
        set1.keys.insert(EqDeleteKey(vec![Some(Datum::long(10))]));
        set1.keys.insert(EqDeleteKey(vec![Some(Datum::long(20))]));

        let mut set2 = EqDeleteSet {
            keys: std::collections::HashSet::new(),
            fields: vec![("id".to_string(), 1)],
        };
        set2.keys.insert(EqDeleteKey(vec![Some(Datum::long(30))]));

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-1.parquet", rx1);
        tx1.send(Arc::new(set1)).unwrap();

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-2.parquet", rx2);
        tx2.send(Arc::new(set2)).unwrap();

        // Small delay to allow the spawned tasks to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let result = filter.build_equality_delete_sets(&task).await;
        assert!(result.is_ok());

        let eq_sets = result.unwrap();
        // Same equality_ids → unioned into one set
        assert_eq!(eq_sets.len(), 1);
        let eq_set = &eq_sets[0];
        // Union of {10, 20} and {30} should contain all three
        assert_eq!(eq_set.keys.len(), 3);
        assert!(
            eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::long(10))]))
        );
        assert!(
            eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::long(20))]))
        );
        assert!(
            eq_set
                .keys
                .contains(&EqDeleteKey(vec![Some(Datum::long(30))]))
        );
    }

    /// Delete files with different equality_ids must NOT be unioned — they
    /// produce separate sets, each applied independently.
    #[tokio::test]
    async fn test_build_equality_delete_sets_different_equality_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let task = FileScanTask {
            file_size_in_bytes: 0,
            start: 0,
            length: 0,
            record_count: None,
            data_file_path: "data.parquet".to_string(),
            data_file_format: crate::spec::DataFileFormat::Parquet,
            schema: schema.clone(),
            project_field_ids: vec![1, 2],
            predicate: None,
            deletes: vec![
                FileScanTaskDeleteFile {
                    file_path: "eq-del-by-id.parquet".to_string(),
                    file_size_in_bytes: 1,
                    file_type: DataContentType::EqualityDeletes,
                    partition_spec_id: 0,
                    equality_ids: Some(vec![1]),
                },
                FileScanTaskDeleteFile {
                    file_path: "eq-del-by-name.parquet".to_string(),
                    file_size_in_bytes: 1,
                    file_type: DataContentType::EqualityDeletes,
                    partition_spec_id: 0,
                    equality_ids: Some(vec![2]),
                },
            ],
            partition: None,
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let filter = DeleteFilter::default();

        // Delete file 1: delete by id
        let mut set_by_id = EqDeleteSet {
            keys: std::collections::HashSet::new(),
            fields: vec![("id".to_string(), 1)],
        };
        set_by_id
            .keys
            .insert(EqDeleteKey(vec![Some(Datum::long(10))]));

        // Delete file 2: delete by name
        let mut set_by_name = EqDeleteSet {
            keys: std::collections::HashSet::new(),
            fields: vec![("name".to_string(), 2)],
        };
        set_by_name
            .keys
            .insert(EqDeleteKey(vec![Some(Datum::string("alice"))]));

        let (tx1, rx1) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-by-id.parquet", rx1);
        tx1.send(Arc::new(set_by_id)).unwrap();

        let (tx2, rx2) = tokio::sync::oneshot::channel();
        filter.insert_equality_delete("eq-del-by-name.parquet", rx2);
        tx2.send(Arc::new(set_by_name)).unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let eq_sets = filter
            .build_equality_delete_sets(&task)
            .await
            .expect("should succeed");

        // Different equality_ids → two separate sets, NOT unioned
        assert_eq!(
            eq_sets.len(),
            2,
            "Delete files with different equality_ids must produce separate sets"
        );

        // Each set should have exactly one key
        let key_counts: Vec<usize> = eq_sets.iter().map(|s| s.keys.len()).collect();
        assert!(key_counts.contains(&1));
    }
}
