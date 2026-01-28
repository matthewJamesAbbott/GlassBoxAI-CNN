/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 */

pub mod cnn;
mod cxx_qt_bridge;

// Kani verification tests (CISA Hardening)
#[cfg(kani)]
mod kani_tests;

use cxx_qt_lib::{QGuiApplication, QQmlApplicationEngine, QUrl};

fn main() {
    let mut app = QGuiApplication::new();
    let mut engine = QQmlApplicationEngine::new();

    engine.pin_mut().load(&QUrl::from("qrc:/qt/qml/com/glassboxai/cnn/qml/main.qml"));

    if let Some(app) = app.as_mut() {
        app.exec();
    }
}
