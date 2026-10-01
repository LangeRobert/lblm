extends Window
## Portrait companion matching the supplied camera-and-conversation design.

const MAX_MESSAGES: int = 100
const MAX_TEXT: int = 16384
var messages := VBoxContainer.new()
var scroll := ScrollContainer.new()
var camera_view := TextureRect.new()
var responses: Dictionary = {}
var follow_latest: bool = true


func configure(visualizer: Window) -> void:
	## Scale the 1080×1920 reference layout while preserving its proportions.
	title = "LBLM · Conversation"
	min_size = Vector2i(360, 640)
	content_scale_size = Vector2i(1080, 1920)
	content_scale_mode = Window.CONTENT_SCALE_MODE_CANVAS_ITEMS
	content_scale_aspect = Window.CONTENT_SCALE_ASPECT_KEEP
	var font := FontFile.new()
	font.data = FileAccess.get_file_as_bytes("res://assets/Poppins-Regular.ttf")
	var skin := Theme.new()
	skin.default_font = font
	skin.default_font_size = 32
	skin.set_color("font_color", "Label", Color.WHITE)
	theme = skin
	var background := ColorRect.new()
	background.color = Color.BLACK
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(background)
	var camera_background := ColorRect.new()
	camera_background.color = Color.WHITE
	camera_background.position = Vector2(54, 47)
	camera_background.size = Vector2(422, 517)
	add_child(camera_background)
	camera_view.position = Vector2(54, 47)
	camera_view.size = Vector2(422, 517)
	camera_view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	camera_view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_COVERED
	add_child(camera_view)
	var heading := Label.new()
	heading.position = Vector2(54, 609)
	heading.size = Vector2(422, 600)
	heading.text = "Large\nbody\nLanguage\nModel"
	heading.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	heading.add_theme_font_size_override("font_size", 80)
	add_child(heading)
	scroll.position = Vector2(540, 0)
	scroll.size = Vector2(540, 1920)
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	add_child(scroll)
	messages.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	messages.add_theme_constant_override("separation", 0)
	scroll.add_child(messages)
	scroll.get_v_scroll_bar().changed.connect(_on_scroll_range_changed)
	close_requested.connect(hide)
	if DisplayServer.get_name() != "headless":
		var area: Rect2i = DisplayServer.screen_get_usable_rect(visualizer.current_screen)
		var height: int = mini(1280, area.size.y - 64)
		var companion_width: int = roundi(height * 1080.0 / 1920.0)
		var visualizer_width: int = mini(1100, area.size.x - companion_width - 48)
		visualizer.size = Vector2i(visualizer_width, mini(760, height))
		visualizer.position = area.position + Vector2i(16, 32)
		size = Vector2i(companion_width, height)
		position = visualizer.position + Vector2i(visualizer_width + 16, 0)
		show()


func add_message(text: String, system_reply: bool) -> Label:
	## Append plain white user text or gray system text, bounding retained entries.
	var label := Label.new()
	label.text = text.left(MAX_TEXT)
	label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	label.add_theme_color_override("font_color", Color("666666") if system_reply else Color.WHITE)
	messages.add_child(label)
	while messages.get_child_count() > MAX_MESSAGES:
		var oldest: Node = messages.get_child(0)
		for response_id: Variant in responses.keys():
			if responses[response_id] == oldest:
				responses.erase(response_id)
		messages.remove_child(oldest)
		oldest.queue_free()
	return label


func receive_event(event: Dictionary) -> void:
	## Stream replies into their gray row without headers or status overlays.
	var bar: VScrollBar = scroll.get_v_scroll_bar()
	follow_latest = bar.value >= bar.max_value - bar.page - 24.0
	var text: String = str(event.get("text", "")).left(MAX_TEXT)
	var response_id: String = str(event.get("response_id", ""))
	match str(event.get("kind", "")):
		"observation":
			add_message(text, false)
		"text":
			if not responses.has(response_id):
				responses[response_id] = add_message("", true)
			var label: Label = responses[response_id]
			label.text = (label.text + text).left(MAX_TEXT)
		"response_done":
			responses.erase(response_id)


func receive_camera(jpeg: String) -> void:
	## Replace the current camera texture from a bounded local JPEG payload.
	var bytes: PackedByteArray = Marshalls.base64_to_raw(jpeg)
	if bytes.size() > 196608:
		return
	var image := Image.new()
	if image.load_jpg_from_buffer(bytes) != OK or image.get_width() > 480 or image.get_height() > 480:
		return
	if camera_view.texture == null:
		camera_view.texture = ImageTexture.create_from_image(image)
	else:
		var texture: ImageTexture = camera_view.texture
		if texture.get_size() == Vector2(image.get_size()):
			texture.update(image)
		else:
			texture.set_image(image)


func _on_scroll_range_changed() -> void:
	## Follow new text only when the reader was already near the bottom.
	if follow_latest:
		scroll.set_deferred("scroll_vertical", int(scroll.get_v_scroll_bar().max_value))
