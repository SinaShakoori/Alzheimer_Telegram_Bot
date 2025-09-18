import os
from datetime import datetime
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

LABELS = {
    0: "Mild_Demented",
    1: "Moderate_Demented",
    2: "Non_Demented",
    3: "Very_Mild_Demented"
}

MODEL_PATH = "model/alzheimer_cnn_model.keras"
model = load_model(MODEL_PATH)

IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    arr = preprocess_image(image_path)
    preds = model.predict(arr)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds)) * 100
    return LABELS[class_id], confidence

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["📷 ارسال عکس"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "سلام 👋\nیک عکس MRI ارسال کنید تا نتیجه پیش‌بینی نمایش داده شود.",
        reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text == "📷 ارسال عکس":
        await update.message.reply_text("لطفاً عکس MRI خود را ارسال کنید.")
    else:
        await update.message.reply_text("فقط عکس ارسال کنید.")

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    photo = await update.message.photo[-1].get_file()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path = os.path.join(DATA_DIR, f"user_{user.id}_{timestamp}.jpg")
    await photo.download_to_drive(img_path)

    label, confidence = predict_image(img_path)

    msg = f"🧠 نتیجه: {label}\n📊 اعتماد: {confidence:.2f}%"
    await update.message.reply_text(msg)

def main():
    app = Application.builder().token("Your Token").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, button_handler))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    app.run_polling()

if __name__ == "__main__":
    main()









    