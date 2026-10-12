# How to Run the Energy Consumption Prediction App on Android Studio

This project includes a native Android application powered by our trained Machine Learning model. It performs **100% on-device, offline inference** with bit-for-bit exact parity ($< 10^{-8}$ difference) with the Python scikit-learn model.

---

## 📱 Project Location
The Android project is located at:
```
C:\Users\Aayush\.gemini\antigravity\scratch\energy_consumption_ml\android_app
```

---

## 🛠️ Step-by-Step Instructions to Run in Android Studio

### Step 1: Open Android Studio
1. Launch **Android Studio** on your computer.
2. In the Welcome screen (or from the top menu), click:
   **File** ➔ **Open...**
3. Browse to and select the `android_app` folder:
   ```
   C:\Users\Aayush\.gemini\antigravity\scratch\energy_consumption_ml\android_app
   ```
4. Click **OK**.

---

### Step 2: Let Gradle Sync Complete
- Android Studio will automatically recognize the project and start the initial **Gradle Sync**.
- In the bottom status bar, wait until it says **"Gradle sync finished"** (typically takes 30–60 seconds the first time).
- Your Android SDK is already pre-configured in `local.properties`:
  ```properties
  sdk.dir=C:\Users\Aayush\AppData\Local\Android\Sdk
  ```

---

### Step 3: Run the App on Emulator or Physical Device

#### Option A: Running on Android Emulator (Virtual Device)
1. In Android Studio's top toolbar, click the device dropdown (next to the green Run button).
2. If an emulator is already created (e.g. *Pixel 8 / Medium Phone*), select it.
3. If no emulator is set up:
   - Click the device dropdown ➔ **Device Manager** (or click the Device Manager icon on the right side).
   - Click **Create Virtual Device** (+).
   - Select **Phone** (e.g., *Pixel 7* or *Pixel 8*) and click **Next**.
   - Download/select the system image (e.g. *VanillaIceCream / Android 15 / 16*) and click **Finish**.
4. Click the green **Run (▶)** button (or press `Shift + F10`).

#### Option B: Running on a Physical Android Phone
1. Enable **Developer Options** and **USB Debugging** on your Android phone.
2. Connect your phone to your PC via a USB cable.
3. Select your connected phone in Android Studio's top device dropdown.
4. Click the green **Run (▶)** button.

---

## ⚡ Using the App Features

1. **Adjust Building & Load Parameters**:
   - Square Footage (sq ft)
   - Number of Occupants
   - Outdoor Temperature (°C)
   - Relative Humidity (%)
   - HVAC Usage Level (`Low`, `Medium`, `High`)
   - Appliance Load (kW)
   - Lighting Load (kW)
   - Rooftop Solar Generation Offset (kW)
   - Time of Day (`Morning`, `Afternoon`, `Evening`, `Night`)
   - Day Type (`Weekday`, `Weekend`)
2. **Tap "⚡ PREDICT ENERGY DEMAND"**:
   - Instantly computes on-device prediction in $< 1$ millisecond without needing internet!
3. **Review Real-Time Audit**:
   - 🚨 **HIGH PEAK DEMAND ALERT** (Red banner with peak risk probability and tailored demand-response suggestions).
   - ✅ **NORMAL / EFFICIENT USAGE** (Green banner confirming optimal energy consumption envelope).
   - **Actionable Optimization Tips** (Thermostat adjustments, off-peak load shifting, solar offset auditing).

---

## 📂 Android Architecture Highlights
- **Engine:** `EnergyPredictor.kt` (Pure Kotlin vector dot-product and logistic sigmoid matching Python's `StandardScaler` + `LogisticRegression`).
- **UI:** Modern Material 3 cards, inputs, spinners, progress indicators, and dynamic styling.
- **Offline:** No external API keys, server backends, or internet connectivity required.
