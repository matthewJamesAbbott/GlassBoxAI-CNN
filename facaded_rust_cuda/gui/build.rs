use cxx_qt_build::{CxxQtBuilder, QmlModule};

fn main() {
    CxxQtBuilder::new()
        .qml_module(QmlModule {
            uri: "com.glassboxai.cnn",
            version_major: 1,
            version_minor: 0,
            rust_files: &["src/cxx_qt_bridge.rs"],
            qml_files: &["qml/main.qml"],
            ..Default::default()
        })
        .build();
}
