import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fer import FER
import tkinter as tk
from tkinter import filedialog

# ─────────────────────────────────────────────
#   EMOTION COLORS
# ─────────────────────────────────────────────
EMOTION_COLORS_BGR = {
    "happy":    (0, 215, 255),
    "sad":      (255, 195, 79),
    "angry":    (80, 83, 239),
    "fear":     (188, 71, 171),
    "disgust":  (102, 187, 106),
    "surprise": (38, 167, 254),
    "neutral":  (176, 190, 197),
}
EMOTION_COLORS_HEX = {
    "happy":    "#FFD700",
    "sad":      "#4FC3F7",
    "angry":    "#EF5350",
    "fear":     "#AB47BC",
    "disgust":  "#66BB6A",
    "surprise": "#FFA726",
    "neutral":  "#B0BEC5",
}

IMAGE_FILES = {
    "happy":    "happy.jpg",
    "sad":      "sad.jpg",
    "angry":    "angry.jpg",
    "fear":     "fear.jpg",
    "disgust":  "disgust.jpg",
    "surprise": "surprise.jpg",
    "neutral":  "neutral.jpg",
}

detector = FER(mtcnn=True)

# ─────────────────────────────────────────────
#   FILE PICKER (Dialog Box)
# ─────────────────────────────────────────────
def pick_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    root.destroy()
    return file_path

# ═════════════════════════════════════════════
#   MODE 1 — SINGLE IMAGE ANALYSIS
# ═════════════════════════════════════════════
def analyze_single_image():
    print("\n📂 Opening file browser... please select an image!")
    image_path = pick_image()

    if not image_path:
        print("❌ No image selected!")
        return

    print(f"✅ Selected: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image!")
        return

    results = detector.detect_emotions(img)
    if not results:
        print("❌ No face detected! Try another image.")
        return

    print(f"✅ {len(results)} face(s) detected!")

    for i, face in enumerate(results):
        x, y, w, h  = face["box"]
        emotions     = face["emotions"]
        top_emotion  = max(emotions, key=emotions.get)
        confidence   = emotions[top_emotion] * 100
        color        = EMOTION_COLORS_BGR.get(top_emotion, (255, 255, 255))

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        label = f"#{i+1} {top_emotion.upper()} {confidence:.0f}%"
        cv2.rectangle(img, (x, y-35), (x+w, y), color, -1)
        cv2.putText(img, label, (x+5, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        print(f"  Face {i+1}: {top_emotion.upper()} ({confidence:.1f}%)")

    # ── Plot ──
    fig = plt.figure(figsize=(14, 6), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3])

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Detected Faces ({len(results)} found)",color="white", fontsize=13, fontweight="bold")
    ax_img.axis("off")

    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor("#16213e")
    emotions    = results[0]["emotions"]
    top_emotion = max(emotions, key=emotions.get)
    sorted_emo  = sorted(emotions.items(), key=lambda x: x[1])
    labels = [e[0] for e in sorted_emo]
    values = [e[1] * 100 for e in sorted_emo]
    colors = [EMOTION_COLORS_HEX.get(l, "#999") for l in labels]
    bars   = ax_bar.barh(labels, values, color=colors, height=0.6)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)", color="#aaa")
    ax_bar.set_title(f"Emotion Scores - Face 1  |  Top: {top_emotion.upper()}",color="white", fontsize=12, fontweight="bold")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[:].set_visible(False)
    for bar, val in zip(bars, values):
        if val > 1:
            ax_bar.text(val + 1, bar.get_y() + bar.get_height()/2,f"{val:.1f}%", va="center", color="white", fontsize=9)

    plt.tight_layout()
    out = os.path.splitext(image_path)[0] + "_result.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    print(f"  💾 Result saved: {out}")
    plt.show()

# ═════════════════════════════════════════════
#   MODE 2 — COMPARISON CHART (ALL 7 EMOTIONS)
# ═════════════════════════════════════════════
def comparison_chart():
    print("\n📊 Select all 7 emotion images one by one!\n")
    all_emo = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    results = {}

    for emotion in ["happy", "sad", "angry", "fear", "disgust", "surprise", "neutral"]:
        print(f"  📂 Select image for: {emotion.upper()}")
        path = pick_image()
        if not path:
            print(f"  ⚠️  Skipped {emotion}")
            continue
        img = cv2.imread(path)
        det = detector.detect_emotions(img)
        if det:
            scores = det[0]["emotions"]
            top    = max(scores, key=scores.get)
            print(f"  ✅ {emotion.upper():10s} → {top.upper()} ({scores[top]*100:.1f}%)")
            results[emotion] = scores
        else:
            print(f"  ❌ {emotion.upper():10s} → No face detected")

    if not results:
        print("❌ No results to plot!")
        return

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Emotion Detection — All 7 Emotions Comparison",fontsize=20, fontweight="bold", color="white")

    axes_flat = axes.flatten()
    for idx, (input_emo, scores) in enumerate(results.items()):
        ax    = axes_flat[idx]
        ax.set_facecolor("#16213e")
        vals  = [scores.get(e, 0) * 100 for e in all_emo]
        bclrs = [EMOTION_COLORS_HEX.get(e, "#999") for e in all_emo]
        bars  = ax.barh(all_emo, vals, color=bclrs, edgecolor="none", height=0.6)
        top   = max(scores, key=scores.get)
        ax.set_xlim(0, 100)
        ax.set_title(f"Input: {input_emo.upper()}\nTop: {top.upper()} ({scores[top]*100:.0f}%)",color=EMOTION_COLORS_HEX.get(input_emo, "white"),fontsize=11, fontweight="bold")
        ax.tick_params(colors="white", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.set_xlabel("Confidence (%)", color="#aaa", fontsize=9)
        for bar, val in zip(bars, vals):
            if val > 2:
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,f"{val:.0f}%", va="center", color="white", fontsize=8)

    for i in range(len(results), 8):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plt.savefig("emotion_comparison_chart.png", dpi=150,
                bbox_inches="tight", facecolor="#1a1a2e")
    print("\n✅ Comparison chart saved: emotion_comparison_chart.png")
    plt.show()

# ═════════════════════════════════════════════
#   MODE 3 — LIVE WEBCAM
# ═════════════════════════════════════════════
def live_webcam():
    print("\n🎥 Starting Live Webcam... Press Q to quit\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_emotions(frame)
        for face in results:
            x, y, w, h  = face["box"]
            emotions     = face["emotions"]
            top_emotion  = max(emotions, key=emotions.get)
            confidence   = emotions[top_emotion] * 100
            color        = EMOTION_COLORS_BGR.get(top_emotion, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{top_emotion.upper()}  {confidence:.1f}%"
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
            cv2.putText(frame, label, (x+5, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            bar_x = min(x + w + 10, frame.shape[1] - 220)
            for i, (emo, score) in enumerate(sorted(emotions.items(),key=lambda x: -x[1])):
                bar_y   = y + i * 28
                bar_len = int(score * 150)
                emo_clr = EMOTION_COLORS_BGR.get(emo, (200, 200, 200))
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_len, bar_y + 20), emo_clr, -1)
                cv2.putText(frame,
                            f"{emo}: {score*100:.0f}%",(bar_x + bar_len + 5, bar_y + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(frame, "Live Emotion Detection | Press Q to Quit",(10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        cv2.imshow("Emotion Detection — Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam closed.")

# ═════════════════════════════════════════════
#   MAIN MENU
# ═════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 52)
    print("    🧠 FACIAL EMOTION RECOGNITION SYSTEM")
    print("=" * 52)
    print()
    print("  1️⃣   Single Image Analysis   (File Browser)")
    print("  2️⃣   All 7 Emotions Comparison Chart")
    print("  3️⃣   Live Webcam Detection")
    print("  4️⃣   Exit")
    print()

    while True:
        choice = input("👉 Enter your choice (1/2/3/4): ").strip()

        if choice == "1":
            analyze_single_image()

        elif choice == "2":
            comparison_chart()

        elif choice == "3":
            live_webcam()

        elif choice == "4":
            print("\n👋 Bye! Come Again!\n")
            break

        else:
            print("  ⚠️  Invalid choice! Enter 1, 2, 3 or 4.")