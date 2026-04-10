//! Utility functions for the markdown renderer.

/// Terminal theme mode (dark or light).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeMode {
    /// Dark terminal background.
    Dark,
    /// Light terminal background.
    #[allow(dead_code)]
    Light,
}

/// Detects the terminal theme mode (dark or light).
pub fn detect_theme_mode() -> ThemeMode {
    #[cfg(test)]
    {
        return ThemeMode::Dark;
    }
    
    #[cfg(not(test))]
    {
        use terminal_colorsaurus::{QueryOptions, ThemeMode as ColorsaurusThemeMode, theme_mode};

        match theme_mode(QueryOptions::default()) {
            Ok(ColorsaurusThemeMode::Light) => ThemeMode::Light,
            Ok(ColorsaurusThemeMode::Dark) | Err(_) => ThemeMode::Dark,
        }
    }
}
