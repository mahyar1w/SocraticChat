mod bench;
mod cli;
mod llama;
mod llm;
mod store;
mod theme;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Instant;

use eframe::egui::{self, Color32, CornerRadius, Frame, Margin, RichText, Stroke, Vec2};

use crate::llama::{LlamaEngine, find_default_gguf, find_llama_server};
use crate::llm::{Message, RequestConfig, generate_reply};
use crate::store::{Conversation, PersistedState, load_state, save_state};
use crate::theme::{Palette, apply_theme};

const MAX_READING_WIDTH: f32 = 680.0;

const STARTERS: &[(&str, &str)] = &[
    ("Casual chat", "Hey! How's your day going?"),
    ("Socratic CS", "Why is binary search O(log n) instead of O(n)?"),
    ("Recommendations", "Recommend 2 chill rainy-night movies"),
    ("Socratic Science", "Why does ice float on water?"),
];

enum EngineState {
    Starting,
    Ready(LlamaEngine),
    Error(String),
}

struct ChatApp {
    state: PersistedState,
    engine: EngineState,
    engine_rx: Option<Receiver<Result<LlamaEngine, String>>>,
    reply_rx: Option<(u64, Receiver<Result<String, String>>)>,
    input: String,
    error_banner: Option<String>,
    show_settings: bool,
    scroll_to_bottom: bool,
    copied_msg: Option<(usize, Instant)>,
}

impl ChatApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let state = load_state();
        apply_theme(&cc.egui_ctx, state.config.dark_mode);
        let mut app = Self {
            state,
            engine: EngineState::Starting,
            engine_rx: None,
            reply_rx: None,
            input: String::new(),
            error_banner: None,
            show_settings: false,
            scroll_to_bottom: false,
            copied_msg: None,
        };
        app.start_engine(cc.egui_ctx.clone());
        app
    }

    fn start_engine(&mut self, ctx: egui::Context) {
        self.engine = EngineState::Starting;
        self.error_banner = None;

        let server_bin = find_llama_server(&self.state.config.llama_server_path)
            .unwrap_or_else(|| PathBuf::from(&self.state.config.llama_server_path));
        let gguf_path = find_default_gguf(&self.state.config.gguf_path)
            .unwrap_or_else(|| PathBuf::from(&self.state.config.gguf_path));

        let (tx, rx) = mpsc::channel();
        self.engine_rx = Some(rx);
        thread::spawn(move || {
            let res = LlamaEngine::start(&server_bin, &gguf_path);
            let _ = tx.send(res);
            ctx.request_repaint();
        });
    }

    fn active_conversation(&self) -> &Conversation {
        self.state
            .conversations
            .iter()
            .find(|c| c.id == self.state.active_id)
            .unwrap_or(&self.state.conversations[0])
    }

    fn active_conversation_mut(&mut self) -> &mut Conversation {
        let active = self.state.active_id;
        if let Some(pos) = self.state.conversations.iter().position(|c| c.id == active) {
            &mut self.state.conversations[pos]
        } else {
            &mut self.state.conversations[0]
        }
    }

    fn new_chat(&mut self) {
        let id = self.state.next_id;
        self.state.next_id += 1;
        self.state.conversations.insert(
            0,
            Conversation {
                id,
                title: "New Chat".to_string(),
                messages: Vec::new(),
            },
        );
        self.state.active_id = id;
        self.error_banner = None;
        self.copied_msg = None;
        save_state(&self.state);
    }

    fn delete_chat(&mut self, id: u64) {
        self.state.conversations.retain(|c| c.id != id);
        if self.state.conversations.is_empty() {
            self.new_chat();
        } else if self.state.active_id == id {
            self.state.active_id = self.state.conversations[0].id;
            self.copied_msg = None;
            save_state(&self.state);
        } else {
            save_state(&self.state);
        }
    }

    fn send_prompt(&mut self, text: String, ctx: egui::Context) {
        let trimmed = text.trim().to_string();
        if trimmed.is_empty() || self.reply_rx.is_some() {
            return;
        }
        let student_url = match &self.engine {
            EngineState::Ready(eng) => eng.base_url().to_string(),
            EngineState::Starting => {
                self.error_banner = Some("Local llama.cpp model is still loading…".to_string());
                return;
            }
            EngineState::Error(e) => {
                self.error_banner = Some(format!("Local llama.cpp engine is not running: {e}"));
                return;
            }
        };

        self.error_banner = None;
        let conv_id = self.state.active_id;
        {
            let conv = self.active_conversation_mut();
            if conv.messages.is_empty() {
                conv.title = make_title(&trimmed);
            }
            conv.messages.push(Message {
                role: "user".to_string(),
                content: trimmed,
            });
        }
        save_state(&self.state);
        self.scroll_to_bottom = true;

        let history = self.active_conversation().messages.clone();
        let cfg = RequestConfig {
            use_teacher: self.state.config.use_teacher,
            teacher_url: self.state.config.teacher_url.clone(),
            teacher_model: self.state.config.teacher_model.clone(),
            teacher_key: self.state.config.teacher_key.clone(),
            student_url,
            student_model: "local-socratic".to_string(),
            student_key: String::new(),
        };

        let (tx, rx) = mpsc::channel();
        self.reply_rx = Some((conv_id, rx));
        thread::spawn(move || {
            let res = generate_reply(&history, &cfg);
            let _ = tx.send(res);
            ctx.request_repaint();
        });
    }

    fn poll_background(&mut self) {
        if let Some(rx) = &self.engine_rx {
            match rx.try_recv() {
                Ok(res) => {
                    self.engine_rx = None;
                    self.engine = match res {
                        Ok(eng) => EngineState::Ready(eng),
                        Err(e) => {
                            self.error_banner = Some(e.clone());
                            EngineState::Error(e)
                        }
                    };
                }
                Err(TryRecvError::Disconnected) => {
                    self.engine_rx = None;
                    let msg = "Engine startup thread exited unexpectedly".to_string();
                    self.error_banner = Some(msg.clone());
                    self.engine = EngineState::Error(msg);
                }
                Err(TryRecvError::Empty) => {}
            }
        }

        if let Some((conv_id, rx)) = &self.reply_rx {
            let conv_id = *conv_id;
            match rx.try_recv() {
                Ok(res) => {
                    self.reply_rx = None;
                    match res {
                        Ok(reply) => {
                            if let Some(conv) =
                                self.state.conversations.iter_mut().find(|c| c.id == conv_id)
                            {
                                conv.messages.push(Message {
                                    role: "assistant".to_string(),
                                    content: reply,
                                });
                            }
                            if conv_id == self.state.active_id {
                                self.scroll_to_bottom = true;
                            }
                            save_state(&self.state);
                        }
                        Err(err) => {
                            if let Some(conv) =
                                self.state.conversations.iter_mut().find(|c| c.id == conv_id)
                                && conv.messages.last().is_some_and(|m| m.role == "user")
                            {
                                conv.messages.pop();
                            }
                            self.error_banner = Some(err);
                            save_state(&self.state);
                        }
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    self.reply_rx = None;
                    if let Some(conv) =
                        self.state.conversations.iter_mut().find(|c| c.id == conv_id)
                        && conv.messages.last().is_some_and(|m| m.role == "user")
                    {
                        conv.messages.pop();
                    }
                    self.error_banner = Some("Reply worker exited unexpectedly".to_string());
                    save_state(&self.state);
                }
                Err(TryRecvError::Empty) => {}
            }
        }

        if let Some((_, at)) = self.copied_msg
            && at.elapsed().as_secs_f32() > 1.5
        {
            self.copied_msg = None;
        }
    }

    fn render_sidebar(&mut self, ctx: &egui::Context, p: Palette) {
        let is_dark = self.state.config.dark_mode;
        let panel_frame = Frame::new()
            .fill(p.sidebar)
            .inner_margin(Margin::symmetric(12, 12))
            .stroke(Stroke::new(1.0_f32, p.border));

        egui::SidePanel::left("sidebar")
            .frame(panel_frame)
            .default_width(244.0)
            .min_width(200.0)
            .max_width(320.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Socratic").strong().size(17.5).color(p.ink));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let btn = egui::Button::new(
                            RichText::new("+ New").color(Color32::WHITE).strong(),
                        )
                        .fill(p.accent)
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::same(7));
                        if ui.add(btn).clicked() {
                            self.new_chat();
                        }
                    });
                });

                ui.add_space(12.0);
                ui.label(RichText::new("Conversations").small().color(p.muted));
                ui.add_space(4.0);

                let mut to_select = None;
                let mut to_delete = None;

                egui::ScrollArea::vertical()
                    .id_salt("conv_list")
                    .max_height((ui.available_height() - 122.0).max(80.0))
                    .show(ui, |ui| {
                        for conv in &self.state.conversations {
                            let selected = conv.id == self.state.active_id;
                            let row_fill = if selected {
                                p.surface
                            } else {
                                Color32::TRANSPARENT
                            };
                            let row_stroke = if selected {
                                Stroke::new(1.0_f32, p.border)
                            } else {
                                Stroke::NONE
                            };
                            Frame::new()
                                .fill(row_fill)
                                .stroke(row_stroke)
                                .corner_radius(CornerRadius::same(8))
                                .inner_margin(Margin::symmetric(8, 5))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        let label_w = (ui.available_width() - 26.0).max(60.0);
                                        let title = RichText::new(&conv.title)
                                            .color(if selected { p.ink } else { p.muted })
                                            .size(13.5);
                                        let resp = ui.add_sized(
                                            [label_w, 22.0],
                                            egui::Label::new(title)
                                                .truncate()
                                                .sense(egui::Sense::click()),
                                        );
                                        if resp.clicked() {
                                            to_select = Some(conv.id);
                                        }
                                        let del = ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("×").size(13.0).color(p.muted),
                                                )
                                                .frame(false),
                                            )
                                            .on_hover_text("Delete conversation");
                                        if del.clicked() {
                                            to_delete = Some(conv.id);
                                        }
                                    });
                                });
                            ui.add_space(2.0);
                        }
                    });

                if let Some(id) = to_select {
                    self.state.active_id = id;
                    self.error_banner = None;
                    self.copied_msg = None;
                    save_state(&self.state);
                }
                if let Some(id) = to_delete {
                    self.delete_chat(id);
                }

                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.horizontal(|ui| {
                        let settings_btn = egui::Button::new("Settings")
                            .fill(if self.show_settings {
                                p.accent_tint
                            } else {
                                p.surface
                            })
                            .stroke(Stroke::new(
                                1.0_f32,
                                if self.show_settings {
                                    p.accent
                                } else {
                                    p.border
                                },
                            ));
                        if ui.add(settings_btn).clicked() {
                            self.show_settings = !self.show_settings;
                        }
                        if ui
                            .add(
                                egui::Button::new(if is_dark { "Light" } else { "Dark" })
                                    .fill(p.surface)
                                    .stroke(Stroke::new(1.0_f32, p.border)),
                            )
                            .clicked()
                        {
                            self.state.config.dark_mode = !self.state.config.dark_mode;
                            apply_theme(ctx, self.state.config.dark_mode);
                            save_state(&self.state);
                        }
                    });

                    ui.add_space(6.0);
                    let mode_label = if self.state.config.use_teacher
                        && !self.state.config.teacher_url.trim().is_empty()
                        && !self.state.config.teacher_model.trim().is_empty()
                    {
                        format!("{} -> 0.8B", self.state.config.teacher_model)
                    } else {
                        "Offline 0.8B".to_string()
                    };
                    ui.label(RichText::new(mode_label).small().color(p.muted));

                    let (dot_color, status_text) = match &self.engine {
                        EngineState::Ready(_) => (p.ok, "Local engine ready"),
                        EngineState::Starting => (p.warn, "Loading 0.8B model…"),
                        EngineState::Error(_) => (p.err_fg, "Engine stopped"),
                    };
                    ui.horizontal(|ui| {
                        let (rect, _) =
                            ui.allocate_exact_size(Vec2::splat(8.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 4.0, dot_color);
                        ui.label(RichText::new(status_text).small().color(p.ink));
                    });
                    ui.add_space(4.0);
                    ui.separator();
                });
            });
    }

    fn render_settings(&mut self, ctx: &egui::Context, p: Palette) {
        if !self.show_settings {
            return;
        }
        let panel_frame = Frame::new()
            .fill(p.surface)
            .inner_margin(Margin::symmetric(20, 14))
            .stroke(Stroke::new(1.0_f32, p.border));

        egui::TopBottomPanel::top("settings_panel")
            .frame(panel_frame)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Settings").strong().size(15.5).color(p.ink));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Close").clicked() {
                            save_state(&self.state);
                            self.show_settings = false;
                        }
                        if ui.button("Reload Engine").clicked() {
                            save_state(&self.state);
                            self.start_engine(ctx.clone());
                        }
                    });
                });
                ui.add_space(10.0);

                let mut should_save = false;
                egui::Grid::new("settings_grid")
                    .num_columns(2)
                    .spacing([14.0, 8.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("GGUF Model").color(p.muted));
                        should_save |= ui
                            .add(
                                egui::TextEdit::singleline(&mut self.state.config.gguf_path)
                                    .desired_width(440.0),
                            )
                            .lost_focus();
                        ui.end_row();

                        ui.label(RichText::new("llama-server Binary").color(p.muted));
                        should_save |= ui
                            .add(
                                egui::TextEdit::singleline(
                                    &mut self.state.config.llama_server_path,
                                )
                                .desired_width(440.0),
                            )
                            .lost_focus();
                        ui.end_row();

                        ui.label(RichText::new("Two-Step Teacher").color(p.muted));
                        should_save |= ui
                            .checkbox(
                                &mut self.state.config.use_teacher,
                                "Query a larger teacher model first (uncheck for 100% offline)",
                            )
                            .changed();
                        ui.end_row();

                        if self.state.config.use_teacher {
                            ui.label(RichText::new("Teacher URL").color(p.muted));
                            should_save |= ui
                                .add(
                                    egui::TextEdit::singleline(&mut self.state.config.teacher_url)
                                        .hint_text("http://localhost:11434/v1 or https://api.openai.com/v1")
                                        .desired_width(440.0),
                                )
                                .lost_focus();
                            ui.end_row();

                            ui.label(RichText::new("Teacher Model").color(p.muted));
                            should_save |= ui
                                .add(
                                    egui::TextEdit::singleline(
                                        &mut self.state.config.teacher_model,
                                    )
                                    .hint_text("qwen3:8b or gpt-4o-mini")
                                    .desired_width(440.0),
                                )
                                .lost_focus();
                            ui.end_row();

                            ui.label(RichText::new("Teacher API Key").color(p.muted));
                            ui.add(
                                egui::TextEdit::singleline(&mut self.state.config.teacher_key)
                                    .password(true)
                                    .hint_text("Kept in memory only, never saved to disk")
                                    .desired_width(440.0),
                            );
                            ui.end_row();
                        }
                    });

                if should_save {
                    save_state(&self.state);
                }
            });
    }

    fn render_composer(&mut self, ctx: &egui::Context, p: Palette) {
        let is_busy = self.reply_rx.is_some();
        let mut send_now = false;

        let panel_frame = Frame::new()
            .fill(p.canvas)
            .inner_margin(Margin::symmetric(20, 12));

        egui::TopBottomPanel::bottom("composer")
            .frame(panel_frame)
            .show(ctx, |ui| {
                let col_w = ui.available_width().min(MAX_READING_WIDTH);
                let side_pad = ((ui.available_width() - col_w) * 0.5).max(0.0);

                ui.horizontal(|ui| {
                    ui.add_space(side_pad);
                    Frame::new()
                        .fill(p.surface)
                        .stroke(Stroke::new(1.0_f32, p.border_strong))
                        .corner_radius(CornerRadius::same(12))
                        .inner_margin(Margin::symmetric(12, 8))
                        .show(ui, |ui| {
                            ui.set_width((col_w - 24.0).max(200.0));
                            ui.horizontal(|ui| {
                                let edit_w = (ui.available_width() - 82.0).max(140.0);
                                let resp = ui.add_sized(
                                    [edit_w, 42.0],
                                    egui::TextEdit::multiline(&mut self.input)
                                        .frame(false)
                                        .hint_text(
                                            "Ask a question or chat… (Enter to send, Shift+Enter for newline)",
                                        )
                                        .desired_rows(2)
                                        .lock_focus(true),
                                );

                                let can_send = !is_busy
                                    && !self.input.trim().is_empty()
                                    && matches!(self.engine, EngineState::Ready(_));

                                if resp.has_focus()
                                    && ui.input(|i| {
                                        i.key_pressed(egui::Key::Enter) && !i.modifiers.shift
                                    })
                                    && can_send
                                {
                                    if self.input.ends_with('\n') {
                                        self.input.pop();
                                    }
                                    send_now = true;
                                }

                                let btn = egui::Button::new(
                                    RichText::new(if is_busy { "Sending" } else { "Send" })
                                        .color(if can_send { Color32::WHITE } else { p.muted })
                                        .strong(),
                                )
                                .fill(if can_send { p.accent } else { p.surface_subtle })
                                .stroke(Stroke::NONE)
                                .corner_radius(CornerRadius::same(8))
                                .min_size(Vec2::new(72.0, 38.0));

                                if ui.add_enabled(can_send, btn).clicked() {
                                    send_now = true;
                                }
                            });
                        });
                });
            });

        if send_now {
            let text = std::mem::take(&mut self.input);
            self.send_prompt(text, ctx.clone());
        }
    }

    fn render_chat(&mut self, ctx: &egui::Context, p: Palette) {
        let active_busy = self
            .reply_rx
            .as_ref()
            .is_some_and(|(id, _)| *id == self.state.active_id);
        let engine_errored = matches!(self.engine, EngineState::Error(_));
        let mut starter_clicked: Option<String> = None;
        let mut dismiss_error = false;
        let mut open_settings = false;
        let mut copied_idx: Option<usize> = None;

        let panel_frame = Frame::new()
            .fill(p.canvas)
            .inner_margin(Margin::symmetric(20, 12));

        egui::CentralPanel::default()
            .frame(panel_frame)
            .show(ctx, |ui| {
                let col_w = ui.available_width().min(MAX_READING_WIDTH);
                let side_pad = ((ui.available_width() - col_w) * 0.5).max(0.0);

                if let Some(err) = &self.error_banner {
                    ui.horizontal(|ui| {
                        ui.add_space(side_pad);
                        Frame::new()
                            .fill(p.err_bg)
                            .stroke(Stroke::new(1.0_f32, p.err_border))
                            .corner_radius(CornerRadius::same(10))
                            .inner_margin(Margin::symmetric(14, 10))
                            .show(ui, |ui| {
                                ui.set_width((col_w - 28.0).max(200.0));
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::Label::new(RichText::new(err).color(p.err_fg)).wrap(),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if ui.small_button("Dismiss").clicked() {
                                                dismiss_error = true;
                                            }
                                            if engine_errored
                                                && ui.small_button("Open Settings").clicked()
                                            {
                                                open_settings = true;
                                            }
                                        },
                                    );
                                });
                            });
                    });
                    ui.add_space(10.0);
                }

                let is_empty = self.active_conversation().messages.is_empty();
                if is_empty && !active_busy {
                    self.render_empty_state(ui, p, col_w, side_pad, &mut starter_clicked);
                } else {
                    let active_id = self.state.active_id;
                    let scroll_to_bottom = self.scroll_to_bottom;
                    let copied_msg = self.copied_msg;

                    egui::ScrollArea::vertical()
                        .id_salt(("chat_scroll", active_id))
                        .auto_shrink([false, false])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            ui.add_space(8.0);
                            let messages = &self.active_conversation().messages;
                            for (idx, msg) in messages.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.add_space(side_pad);
                                    ui.vertical(|ui| {
                                        ui.set_width(col_w);
                                        render_message(ui, p, idx, msg, copied_msg, &mut copied_idx);
                                    });
                                });
                                ui.add_space(14.0);
                            }

                            if active_busy {
                                ui.horizontal(|ui| {
                                    ui.add_space(side_pad);
                                    Frame::new()
                                        .fill(p.surface)
                                        .stroke(Stroke::new(1.0_f32, p.border))
                                        .corner_radius(CornerRadius::same(12))
                                        .inner_margin(Margin::symmetric(14, 10))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.spinner();
                                                ui.label(
                                                    RichText::new("Thinking…").color(p.muted),
                                                );
                                            });
                                        });
                                });
                                ui.add_space(12.0);
                            }

                            if scroll_to_bottom {
                                ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
                            }
                        });
                    self.scroll_to_bottom = false;
                }
            });

        if dismiss_error {
            self.error_banner = None;
        }
        if open_settings {
            self.show_settings = true;
        }
        if let Some(idx) = copied_idx {
            self.copied_msg = Some((idx, Instant::now()));
        }
        if let Some(prompt) = starter_clicked {
            self.send_prompt(prompt, ctx.clone());
        }
    }

    fn render_empty_state(
        &self,
        ui: &mut egui::Ui,
        p: Palette,
        col_w: f32,
        side_pad: f32,
        starter_clicked: &mut Option<String>,
    ) {
        let top_pad = ((ui.available_height() - 280.0) * 0.35).max(24.0);
        ui.add_space(top_pad);
        ui.horizontal(|ui| {
            ui.add_space(side_pad);
            ui.vertical(|ui| {
                ui.set_width(col_w);
                ui.label(
                    RichText::new("What would you like to explore?")
                        .strong()
                        .size(23.0)
                        .color(p.ink),
                );
                ui.add_space(6.0);
                ui.label(
                    RichText::new(
                        "Chat naturally on everyday topics, or work through math, code, and science one step at a time.",
                    )
                    .color(p.muted),
                );
                ui.add_space(20.0);

                let card_w = ((col_w - 12.0) * 0.5).max(180.0);
                egui::Grid::new("starters_grid")
                    .num_columns(2)
                    .spacing([12.0, 12.0])
                    .show(ui, |ui| {
                        for (i, &(label, prompt)) in STARTERS.iter().enumerate() {
                            let resp = Frame::new()
                                .fill(p.surface)
                                .stroke(Stroke::new(1.0_f32, p.border))
                                .corner_radius(CornerRadius::same(10))
                                .inner_margin(Margin::symmetric(14, 12))
                                .show(ui, |ui| {
                                    ui.set_width((card_w - 28.0).max(140.0));
                                    ui.set_min_height(52.0);
                                    ui.label(RichText::new(label).small().strong().color(p.accent));
                                    ui.add_space(3.0);
                                    ui.add(
                                        egui::Label::new(
                                            RichText::new(prompt).size(13.5).color(p.ink),
                                        )
                                        .wrap()
                                        .selectable(false),
                                    );
                                })
                                .response
                                .interact(egui::Sense::click());
                            if resp.hovered() {
                                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                            }
                            if resp.clicked() {
                                *starter_clicked = Some(prompt.to_string());
                            }
                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
            });
        });
    }
}

fn render_message(
    ui: &mut egui::Ui,
    p: Palette,
    idx: usize,
    msg: &Message,
    copied_msg: Option<(usize, Instant)>,
    copied_idx: &mut Option<usize>,
) {
    let max_user_w = (ui.available_width() * 0.80).min(540.0);
    if msg.role == "user" {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
            Frame::new()
                .fill(p.user_bubble)
                .stroke(Stroke::new(1.0_f32, p.user_border))
                .corner_radius(CornerRadius::same(14))
                .inner_margin(Margin::symmetric(15, 11))
                .show(ui, |ui| {
                    ui.set_max_width(max_user_w);
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                        ui.add(
                            egui::Label::new(RichText::new(&msg.content).color(p.user_ink))
                                .wrap()
                                .selectable(true),
                        );
                    });
                });
        });
    } else {
        Frame::new()
            .fill(p.surface)
            .stroke(Stroke::new(1.0_f32, p.border))
            .corner_radius(CornerRadius::same(14))
            .inner_margin(Margin::symmetric(16, 12))
            .show(ui, |ui| {
                ui.set_width((ui.available_width() - 32.0).max(160.0));
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Socratic").small().strong().color(p.accent));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let is_copied = copied_msg.is_some_and(|(i, _)| i == idx);
                        let label = if is_copied { "Copied" } else { "Copy" };
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(label)
                                        .small()
                                        .color(if is_copied { p.ok } else { p.muted }),
                                )
                                .frame(false),
                            )
                            .clicked()
                        {
                            ui.ctx().copy_text(msg.content.clone());
                            *copied_idx = Some(idx);
                        }
                    });
                });
                ui.add_space(4.0);
                ui.add(
                    egui::Label::new(RichText::new(&msg.content).color(p.ink))
                        .wrap()
                        .selectable(true),
                );
            });
    }
}

impl eframe::App for ChatApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_background();
        let p = Palette::for_mode(self.state.config.dark_mode);
        self.render_sidebar(ctx, p);
        self.render_settings(ctx, p);
        self.render_composer(ctx, p);
        self.render_chat(ctx, p);
    }
}

fn make_title(prompt: &str) -> String {
    let clean: String = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = clean.chars();
    let short: String = chars.by_ref().take(34).collect();
    if chars.next().is_some() {
        format!("{short}…")
    } else {
        short
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(cmd) = args.first().map(String::as_str) {
        let code = match cmd {
            "ask" => cli::run_ask(&args[1..]),
            "bench" => bench::run_bench(&args[1..]),
            "-h" | "--help" => {
                println!(
                    "Socratic Chat — local 0.8B Socratic tutor & conversationalist\n\n\
                     Usage:\n  \
                       socratic-chat              Launch desktop GUI\n  \
                       socratic-chat ask [PROMPT] Run CLI chat (replaces socratic_twist.py)\n  \
                       socratic-chat bench        Run Android/Termux llama-bench suite"
                );
                0
            }
            other => {
                eprintln!("Unknown command: {other}. Run `socratic-chat --help`.");
                2
            }
        };
        std::process::exit(code);
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Socratic Chat")
            .with_inner_size([1020.0, 700.0])
            .with_min_inner_size([680.0, 460.0]),
        ..Default::default()
    };
    if let Err(e) = eframe::run_native(
        "Socratic Chat",
        options,
        Box::new(|cc| Ok(Box::new(ChatApp::new(cc)))),
    ) {
        eprintln!("Failed to launch GUI: {e}");
        std::process::exit(1);
    }
}
