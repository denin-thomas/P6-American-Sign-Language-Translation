import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import threading
import queue
from groq import Groq
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, RoundedRectangle, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.properties import StringProperty, ListProperty, BooleanProperty, NumericProperty
from kivy.metrics import dp
from kivy.graphics.texture import Texture

# Initialize Groq client
client = Groq(api_key="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")

# Load your trained model
model = tf.keras.models.load_model("modelAdditionalLayers5block.keras")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A', 'B', ..., 'Z']


def draw_hand_landmarks_on_black(hand_landmarks, image_size=128):
    landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

    x_vals = landmarks[:, 0]
    y_vals = landmarks[:, 1]
    min_x, max_x = np.min(x_vals), np.max(x_vals)
    min_y, max_y = np.min(y_vals), np.max(y_vals)

    width = max_x - min_x
    height = max_y - min_y
    scale = 0.8 * image_size / max(width, height)

    landmarks[:, 0] = (landmarks[:, 0] - min_x) * scale
    landmarks[:, 1] = (landmarks[:, 1] - min_y) * scale

    offset_x = (image_size - width * scale) / 2
    offset_y = (image_size - height * scale) / 2
    landmarks[:, 0] += offset_x
    landmarks[:, 1] += offset_y

    black_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    landmark_points = landmarks.astype(np.int32)

    for connection in mp_hands.HAND_CONNECTIONS:
        start = tuple(landmark_points[connection[0]])
        end = tuple(landmark_points[connection[1]])
        cv2.line(black_img, start, end, (0, 255, 0), 1)

    for point in landmark_points:
        cv2.circle(black_img, tuple(point), 1, (0, 255, 0), -1)

    return black_img


def generate_chat_response(prompt, response_queue):
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        response = chat_completion.choices[0].message.content
        response_queue.put(("assistant", response))
    except Exception as e:
        response_queue.put(("error", f"API Error: {str(e)}"))


class ChatBubble(BoxLayout):
    text = StringProperty("")
    sender = StringProperty("user")
    timestamp = StringProperty("")

    def __init__(self, **kwargs):
        super(ChatBubble, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, None)
        self.height = dp(100)  # Default height, will adjust
        self.padding = dp(10)
        self.spacing = dp(5)

        # Timestamp label
        ts_label = Label(
            text=self.timestamp,
            size_hint=(1, None),
            height=dp(20),
            font_size=dp(12),
            color=(0.7, 0.7, 0.7, 1),
            halign='right'
        )

        # Message content
        content_label = Label(
            text=self.text,
            size_hint=(1, None),
            text_size=(Window.width * 0.5, None),
            halign='left',
            valign='top',
            markup=True
        )
        content_label.bind(texture_size=self.update_height)

        self.add_widget(ts_label)
        self.add_widget(content_label)

        with self.canvas.before:
            if self.sender == "user":
                Color(0.27, 0.27, 0.39, 1)
            else:
                Color(0.31, 0.19, 0.27, 1)
            self.rect = RoundedRectangle(
                radius=[dp(15)],
                pos=self.pos,
                size=self.size
            )

        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update_height(self, instance, size):
        instance.height = size[1]
        self.height = instance.height + dp(40)  # Add padding


class CameraWidget(Image):
    def __init__(self, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        self.sentence = ""
        self.last_added = ""
        self.letter_added_time = 0
        self.letter_delay = 1.2
        self.current_frame_letter = None
        self.consecutive_frame_count = 0
        self.required_consecutive_frames = 5
        self.prediction_text = "Prediction: -"
        self.app = None

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        display_img = np.zeros((480, 640, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                colored_img = draw_hand_landmarks_on_black(hand_landmarks)
                gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
                normalized_img = gray_img.astype("float32") / 255.0
                input_img = np.expand_dims(normalized_img, axis=(0, -1))

                prediction = model.predict(input_img, verbose=0)
                predicted_index = np.argmax(prediction)
                confidence = prediction[0][predicted_index]

                if confidence > 0.8:
                    detected_letter = labels[predicted_index]
                    self.prediction_text = f"Prediction: {detected_letter} ({confidence * 100:.1f}%)"

                    if self.app:
                        self.app.prediction_text = self.prediction_text

                    if detected_letter == self.current_frame_letter:
                        self.consecutive_frame_count += 1
                    else:
                        self.current_frame_letter = detected_letter
                        self.consecutive_frame_count = 1

                    current_time = time.time()
                    if (self.consecutive_frame_count >= self.required_consecutive_frames and
                            (detected_letter != self.last_added or
                             current_time - self.letter_added_time > self.letter_delay)):
                        self.sentence += detected_letter
                        self.last_added = detected_letter
                        self.letter_added_time = current_time

                        if self.app:
                            self.app.sentence = self.sentence

                        self.consecutive_frame_count = 0
                        self.current_frame_letter = None

                else:
                    self.prediction_text = "Prediction: Unsure"
                    if self.app:
                        self.app.prediction_text = self.prediction_text
                    self.consecutive_frame_count = 0
                    self.current_frame_letter = None

                resized_colored_img = cv2.resize(colored_img, (400, 400))
                h, w = resized_colored_img.shape[:2]
                display_img[40:40 + h, 120:120 + w] = resized_colored_img
                cv2.putText(display_img, self.prediction_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                break
        else:
            self.consecutive_frame_count = 0
            self.current_frame_letter = None
            self.prediction_text = "Prediction: -"
            if self.app:
                self.app.prediction_text = self.prediction_text

            cv2.putText(display_img, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        buf = cv2.flip(display_img, 0).tobytes()
        texture = self.texture
        if texture is None or texture.width != display_img.shape[1] or texture.height != display_img.shape[0]:
            self.texture = texture = Texture.create(size=(display_img.shape[1], display_img.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.canvas.ask_update()

    def on_stop(self):
        self.capture.release()


class ASLChatApp(App):
    sentence = StringProperty("")
    prediction_text = StringProperty("Prediction: -")
    thinking = BooleanProperty(False)
    thinking_start_time = NumericProperty(0)
    thinking_text = StringProperty("Thinking")
    thinking_animation_event = None

    def __init__(self, **kwargs):
        super(ASLChatApp, self).__init__(**kwargs)
        self.thinking_animation_event = None

    def start_thinking_animation(self):
        if self.thinking_animation_event is None:
            self.thinking_animation_event = Clock.schedule_interval(self.update_thinking_text, 0.5)

    def stop_thinking_animation(self):
        if self.thinking_animation_event:
            self.thinking_animation_event.cancel()
            self.thinking_animation_event = None
        self.thinking_text = "Thinking"

    def update_thinking_text(self, dt):
        dots = "." * (int((time.time() - self.thinking_start_time) * 2) % 4)
        self.thinking_text = f"Thinking{dots}"

    def on_thinking(self, instance, value):
        if value:
            self.thinking_start_time = time.time()
            self.start_thinking_animation()
        else:
            self.stop_thinking_animation()

    def build(self):
        Window.size = (1000, 700)
        self.title = "ASL Chat Assistant"

        self.root = Builder.load_string("""
BoxLayout:
    orientation: 'horizontal'
    padding: dp(10)
    spacing: dp(10)

    BoxLayout:
        orientation: 'vertical'
        size_hint: 0.4, 1
        spacing: dp(10)
        padding: dp(5)

        BoxLayout:
            orientation: 'vertical'
            size_hint: 1, 0.7
            padding: dp(5)

            Label:
                text: "ASL Landmark Visualization"
                bold: True
                font_size: dp(18)
                color: 0.2, 0.2, 0.2, 1
                size_hint_y: None
                height: dp(30)

            CameraWidget:
                id: camera_widget

            Label:
                text: app.prediction_text
                bold: True
                font_size: dp(16)
                color: 0, 0.5, 0, 1
                size_hint_y: None
                height: dp(30)

        BoxLayout:
            orientation: 'vertical'
            size_hint: 1, 0.3
            padding: dp(5)
            spacing: dp(5)

            Label:
                text: "Your Message"
                bold: True
                font_size: dp(18)
                color: 0.2, 0.2, 0.2, 1
                size_hint_y: None
                height: dp(30)

            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: dp(40)
                padding: dp(5)
                canvas.before:
                    Color:
                        rgba: 0.95, 0.95, 0.95, 1
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [dp(10),]

                Label:
                    id: sentence_label
                    text: app.sentence if app.sentence else "Type using ASL..."
                    font_size: dp(18)
                    color: 0.4, 0.4, 0.4, 1
                    text_size: self.width, None
                    halign: 'left'
                    valign: 'middle'
                    padding_x: dp(10)

            BoxLayout:
                orientation: 'horizontal'
                size_hint_y: None
                height: dp(40)
                spacing: dp(10)

                Button:
                    text: "SPACE"
                    font_size: dp(16)
                    background_normal: ''
                    background_color: 0.9, 0.9, 0.9, 1
                    color: 0.2, 0.2, 0.2, 1
                    on_press: app.add_space()

                Button:
                    text: "CLEAR"
                    font_size: dp(16)
                    background_normal: ''
                    background_color: 0.9, 0.9, 0.9, 1
                    color: 0.8, 0.2, 0.2, 1
                    on_press: app.clear_sentence()

                Button:
                    text: "SEND"
                    font_size: dp(16)
                    background_normal: ''
                    background_color: 0.2, 0.6, 0.2, 1
                    color: 1, 1, 1, 1
                    on_press: app.send_message()

    BoxLayout:
        orientation: 'vertical'
        size_hint: 0.6, 1
        padding: dp(5)
        spacing: dp(5)

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: dp(50)
            padding: dp(5)

            Label:
                text: "ASL Chat Assistant"
                bold: True
                font_size: dp(22)
                color: 0.2, 0.2, 0.2, 1

            Button:
                text: "Clear Chat"
                size_hint_x: None
                width: dp(100)
                background_normal: ''
                background_color: 0.8, 0.2, 0.2, 1
                color: 1, 1, 1, 1
                on_press: app.clear_chat()

        ScrollView:
            id: scroll_view
            scroll_type: ['bars', 'content']
            bar_width: dp(10)

            BoxLayout:
                id: chat_container
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                padding: dp(10)
                spacing: dp(10)

        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: dp(30) if app.thinking else 0
            opacity: 1 if app.thinking else 0

            Label:
                text: app.thinking_text
                font_size: dp(18)
                color: 0.5, 0.8, 1, 1
                padding_x: dp(20)
""")

        self.camera = self.root.ids.camera_widget
        self.camera.app = self
        self.chat_container = self.root.ids.chat_container
        self.scroll_view = self.root.ids.scroll_view
        self.sentence_label = self.root.ids.sentence_label
        self.response_queue = queue.Queue()

        Clock.schedule_interval(self.check_queue, 0.1)
        self.add_message("assistant",
                         "Hello! I'm your ASL assistant. Start signing to build a message, then press SEND when you're ready.",
                         time.time())

        return self.root

    def on_stop(self):
        self.camera.on_stop()
        self.stop_thinking_animation()

    def add_message(self, sender, text, timestamp):
        time_str = time.strftime("%H:%M", time.localtime(timestamp))
        bubble = ChatBubble(
            text=text,
            sender=sender,
            timestamp=time_str
        )
        self.chat_container.add_widget(bubble)
        Clock.schedule_once(self.scroll_to_bottom, 0.1)

    def scroll_to_bottom(self, dt):
        self.scroll_view.scroll_y = 0

    def     check_queue(self, dt):
        try:
            if not self.response_queue.empty():
                role, message = self.response_queue.get()
                if role == "assistant":
                    self.add_message("assistant", message, time.time())
                elif role == "error":
                    self.add_message("system", message, time.time())
                self.thinking = False
        except Exception as e:
            print(f"Error processing queue: {e}")

    def add_space(self):
        """Add space to the current sentence"""
        self.sentence += " "
        self.sentence_label.text = self.sentence if self.sentence else "Type using ASL..."

    def clear_sentence(self):
        """Clear only the current typing sentence"""
        self.sentence = "fuckck"
        self.sentence_label.text = "Type using ASL..."
        # Force update the prediction text
        self.prediction_text = "Prediction: -"

    def clear_chat(self):
        """Clear all messages from the chat container and reset the current sentence"""
        self.chat_container.clear_widgets()
        self.sentence = ""
        self.sentence_label.text = "Type using ASL..."
        self.prediction_text = "Prediction: -"
        self.add_message("assistant",
                        "Chat cleared. Start signing to build a new message.",
                        time.time())

    def send_message(self):
        if not self.sentence.strip():
            return

        # Store message before clearing
        message_text = self.sentence.strip()

        # Add to chat immediately
        self.add_message("user", message_text, time.time())

        # Clear input field immediately after sending
        self.sentence = ""
        self.sentence_label.text = "Type using ASL..."
        self.prediction_text = "Prediction: -"

        # Show thinking indicator
        self.thinking = True
        self.thinking_start_time = time.time()

        # Start response generation in background
        threading.Thread(
            target=generate_chat_response,
            args=(message_text, self.response_queue),
            daemon=True
        ).start()


if __name__ == "__main__":
    ASLChatApp().run()