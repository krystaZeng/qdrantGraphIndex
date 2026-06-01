//! MIRAGE-ANNS vector index.
//!
//! See [`mirage`] for the runtime/build implementation, [`config`] for
//! persistence, and [`refinement_builder`] for the Layer 0 construction
//! algorithm.
//!
//! Public re-exports below are the surface used by `vector_index_base.rs`
//! and `segment_constructor`.

pub mod config;
pub(crate) mod faiss_random;
pub mod mirage;
pub mod refinement_builder;

pub use config::{MIRAGE_INDEX_CONFIG_FILE, MirageGraphConfig};
pub use mirage::{MirageIndex, MirageIndexOpenArgs};
pub use refinement_builder::{RefinementParams, build_layer0};
