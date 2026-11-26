use eframe::egui::{self, Color32, CornerRadius, FontId, Stroke, TextStyle, Vec2};

#[derive(Clone, Copy)]
pub struct Palette {
    pub canvas: Color32,
    pub sidebar: Color32,
    pub surface: Color32,
    pub surface_subtle: Color32,
    pub border: Color32,
    pub border_strong: Color32,
    pub ink: Color32,
    pub muted: Color32,
    pub accent: Color32,
    pub accent_tint: Color32,
    pub user_bubble: Color32,
    pub user_ink: Color32,
    pub user_border: Color32,
    pub ok: Color32,
    pub warn: Color32,
    pub err_bg: Color32,
    pub err_fg: Color32,
    pub err_border: Color32,
}

impl Palette {
    pub fn for_mode(dark: bool) -> Self {
        if dark {
            Self {
                canvas: Color32::from_rgb(21, 20, 18),
                sidebar: Color32::from_rgb(28, 26, 23),
                surface: Color32::from_rgb(35, 32, 28),
                surface_subtle: Color32::from_rgb(31, 29, 25),
                border: Color32::from_rgb(50, 46, 40),
                border_strong: Color32::from_rgb(74, 68, 59),
                ink: Color32::from_rgb(237, 233, 224),
                muted: Color32::from_rgb(158, 151, 140),
                accent: Color32::from_rgb(185, 83, 41),
                accent_tint: Color32::from_rgb(51, 35, 27),
                user_bubble: Color32::from_rgb(44, 40, 34),
                user_ink: Color32::from_rgb(242, 238, 230),
                user_border: Color32::from_rgb(68, 62, 53),
                ok: Color32::from_rgb(76, 168, 115),
                warn: Color32::from_rgb(217, 155, 56),
                err_bg: Color32::from_rgb(45, 25, 23),
                err_fg: Color32::from_rgb(243, 166, 155),
                err_border: Color32::from_rgb(122, 50, 41),
            }
        } else {
            Self {
                canvas: Color32::from_rgb(247, 245, 240),
                sidebar: Color32::from_rgb(239, 236, 228),
                surface: Color32::from_rgb(255, 255, 255),
                surface_subtle: Color32::from_rgb(242, 239, 232),
                border: Color32::from_rgb(221, 215, 203),
                border_strong: Color32::from_rgb(196, 188, 173),
                ink: Color32::from_rgb(31, 29, 26),
                muted: Color32::from_rgb(99, 93, 83),
                accent: Color32::from_rgb(173, 78, 39),
                accent_tint: Color32::from_rgb(246, 233, 226),
                user_bubble: Color32::from_rgb(38, 35, 31),
                user_ink: Color32::from_rgb(250, 248, 244),
                user_border: Color32::from_rgb(38, 35, 31),
                ok: Color32::from_rgb(46, 125, 79),
                warn: Color32::from_rgb(180, 105, 14),
                err_bg: Color32::from_rgb(253, 242, 240),
                err_fg: Color32::from_rgb(166, 44, 30),
                err_border: Color32::from_rgb(229, 168, 158),
            }
        }
    }
}

pub fn apply_theme(ctx: &egui::Context, dark: bool) {
    let p = Palette::for_mode(dark);
    let mut visuals = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };

    visuals.panel_fill = p.canvas;
    visuals.window_fill = p.surface;
    visuals.extreme_bg_color = p.surface;
    visuals.faint_bg_color = p.surface_subtle;
    visuals.override_text_color = Some(p.ink);
    visuals.hyperlink_color = p.accent;
    visuals.selection.bg_fill = p.accent_tint;
    visuals.selection.stroke = Stroke::new(1.2_f32, p.accent);

    visuals.widgets.noninteractive.bg_fill = p.canvas;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, p.border);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, p.ink);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(8);

    visuals.widgets.inactive.bg_fill = p.surface_subtle;
    visuals.widgets.inactive.weak_bg_fill = p.surface_subtle;
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, p.border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, p.ink);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(8);

    visuals.widgets.hovered.bg_fill = p.surface;
    visuals.widgets.hovered.weak_bg_fill = p.surface;
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, p.border_strong);
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, p.ink);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(8);

    visuals.widgets.active.bg_fill = p.accent_tint;
    visuals.widgets.active.weak_bg_fill = p.accent_tint;
    visuals.widgets.active.bg_stroke = Stroke::new(1.2_f32, p.accent);
    visuals.widgets.active.fg_stroke = Stroke::new(1.0_f32, p.ink);
    visuals.widgets.active.corner_radius = CornerRadius::same(8);

    ctx.set_visuals(visuals);

    ctx.style_mut(|style| {
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(12.0, 6.0);
        style.text_styles.insert(TextStyle::Body, FontId::proportional(14.5));
        style.text_styles.insert(TextStyle::Button, FontId::proportional(13.5));
        style.text_styles.insert(TextStyle::Small, FontId::proportional(12.0));
        style.text_styles.insert(TextStyle::Heading, FontId::proportional(21.0));
    });
}
