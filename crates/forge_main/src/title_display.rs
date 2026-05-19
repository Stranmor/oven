use std::fmt;

use chrono::Local;
use colored::Colorize;
use forge_domain::{Category, TitleFormat};

/// Implementation of Display for TitleFormat in the presentation layer
pub struct TitleDisplay {
    inner: TitleFormat,
    with_colors: bool,
}

impl TitleDisplay {
    pub fn new(title: TitleFormat) -> Self {
        Self { inner: title, with_colors: true }
    }

    pub fn with_colors(mut self, with_colors: bool) -> Self {
        self.with_colors = with_colors;
        self
    }

    fn format_multiline_sub_title(&self, sub_title: &str) -> String {
        let mut lines = sub_title.lines();
        let Some(first) = lines.next() else {
            return String::new();
        };
        let mut output = String::new();
        if self.with_colors {
            output.push_str(&format!(" {}", first.dimmed()));
        } else {
            output.push_str(&format!(" {first}"));
        }
        for line in lines {
            if self.with_colors {
                output.push_str(&format!("\n  {}", line.dimmed()));
            } else {
                output.push_str(&format!("\n  {line}"));
            }
        }
        output
    }

    fn format_with_colors(&self) -> String {
        let mut buf = String::new();

        let icon = match self.inner.category {
            Category::Action => "●".yellow(),
            Category::Info => "●".white(),
            Category::Debug => "●".cyan(),
            Category::Error => "●".red(),
            Category::Completion => "●".yellow(),
            Category::Warning => "⚠️".bright_yellow(),
        };

        buf.push_str(format!("{icon} ").as_str());

        let local_time: chrono::DateTime<Local> = self.inner.timestamp.into();
        let timestamp_str = format!("[{}] ", local_time.format("%H:%M:%S"));
        buf.push_str(timestamp_str.dimmed().to_string().as_str());

        let title = match self.inner.category {
            Category::Action => self.inner.title.white(),
            Category::Info => self.inner.title.white(),
            Category::Debug => self.inner.title.dimmed(),
            Category::Error => format!("{} {}", "ERROR:".bold(), self.inner.title).red(),
            Category::Completion => self.inner.title.white().bold(),
            Category::Warning => {
                format!("{} {}", "WARNING:".bold(), self.inner.title).bright_yellow()
            }
        };

        buf.push_str(title.to_string().as_str());

        if let Some(ref sub_title) = self.inner.sub_title {
            buf.push_str(&self.format_multiline_sub_title(sub_title));
        }

        buf
    }

    fn format_plain(&self) -> String {
        let mut buf = String::new();

        buf.push_str("● ");

        let local_time: chrono::DateTime<Local> = self.inner.timestamp.into();
        let timestamp_str = format!("[{}] ", local_time.format("%H:%M:%S"));
        buf.push_str(&timestamp_str);

        buf.push_str(&self.inner.title);

        if let Some(ref sub_title) = self.inner.sub_title {
            buf.push_str(&self.format_multiline_sub_title(sub_title));
        }

        buf
    }
}

impl fmt::Display for TitleDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.with_colors {
            write!(f, "{}", self.format_with_colors())
        } else {
            write!(f, "{}", self.format_plain())
        }
    }
}

/// Extension trait to easily convert TitleFormat to displayable form
pub trait TitleDisplayExt {
    fn display(self) -> TitleDisplay;
    fn display_with_colors(self, with_colors: bool) -> TitleDisplay;
}

impl TitleDisplayExt for TitleFormat {
    fn display(self) -> TitleDisplay {
        TitleDisplay::new(self)
    }

    fn display_with_colors(self, with_colors: bool) -> TitleDisplay {
        TitleDisplay::new(self).with_colors(with_colors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_title_display_preserves_multiline_subtitle_plain() {
        let fixture = TitleFormat::debug("Task").sub_title("first line\nsecond line");

        let actual = fixture.display_with_colors(false).to_string();
        let expected_suffix = "Task first line\n  second line";
        assert!(actual.ends_with(expected_suffix));
    }

    #[test]
    fn test_title_display_preserves_single_line_subtitle_plain() {
        let fixture = TitleFormat::debug("Task").sub_title("single line");

        let actual = fixture.display_with_colors(false).to_string();
        let expected_suffix = "Task single line";
        assert!(actual.ends_with(expected_suffix));
    }
}
