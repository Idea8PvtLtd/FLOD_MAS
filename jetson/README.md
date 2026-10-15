# FLOD_MAS on the Jetson Nano 4GB

For the **original Jetson Nano 4GB** with **JetPack 4.6.x** (L4T R32.7, Ubuntu 18.04).
The app runs on Python 3.8, built by the setup script, and keeps itself updated from the
`dev-jetson-lm` branch through the same updater the Windows PCs use.

## 1. Prepare the Jetson
1. Flash JetPack 4.6.x (SD card image or SDK Manager) and finish the Ubuntu first-boot setup.
2. Check the release:
   ```bash
   cat /etc/nv_tegra_release     # must show R32 (e.g. R32 REVISION: 7.x)
   ```
3. Turn on automatic login: **Settings → Users → Unlock → Automatic Login: ON**.
   AppStarter starts when the desktop session starts.
4. Plug in the camera and check that it offers MJPG at full size:
   ```bash
   v4l2-ctl --list-formats-ext -d /dev/video0 | grep -A2 MJPG
   ```

## 2. Install (once, about an hour, mostly building Python 3.8)
```bash
git config --global credential.helper store
git clone -b dev-jetson-lm https://github.com/Idea8PvtLtd/FLOD_MAS.git ~/FLOD_MAS_setup
bash ~/FLOD_MAS_setup/jetson/setup_jetson.sh
```
The clone asks for a GitHub user name and a personal access token with read access to
the repository. The credential helper stores them, so the starter can fetch updates later
without a prompt.

The setup script:
- installs build tools;
- builds Python 3.8 with tkinter into `/usr/local/bin/python3.8`;
- adds a 4 GB swap file;
- sets full power (`nvpmodel -m 0` and `jetson_clocks` at every boot);
- adds the user to the `video` and `dialout` groups;
- creates `~/AppStarter` and registers it to start at login.

Reboot afterwards.

## 3. First start
At the first start, AppStarter clones the repository and unpacks the newest
`dev-jetson-lm` build into `~/AppStarter/releases/<commit>`. It then creates
`~/AppStarter/venv` with Python 3.8, installs `requirements-jetson.txt` (TensorFlow,
MediaPipe and so on; allow 20–40 minutes) and starts `main_1366.py`. Later starts only
download what changed.

To start it by hand and watch the log:
```bash
~/AppStarter/run_appstarter.sh &
tail -f ~/AppStarter/starter.log
```

`~/AppStarter/updater.json` chooses the branch, Python version and requirements file.

## Calibration and data
Settings, calibration and captures live in `~/Documents/data files/`, the same layout as on
the PCs. Calibrate the camera on the Jetson itself (Settings → Camera setup), or copy
`Files/` from a PC that uses the same camera at the same resolution and rotation.

## Speed and memory
- Everything runs on the CPU for now, so expect several seconds to tens of seconds per glue
  line. Watch memory and load with `tegrastats`.
- `auto_trigger.max_queued_captures` is 1 in `config.json`: a hand cycle made while one
  capture is already waiting is not captured, and the status line says so. Raise it only if
  `tegrastats` shows spare memory.
- **Planned next step:** run the models on the GPU with NVIDIA's `onnxruntime-gpu` 1.11 for
  JetPack 4.6 and `local_inspection.execution_provider: "cuda"` / `"tensorrt"`. Those
  builds need the models converted to ONNX opset 16.

## Troubleshooting
- **App does not start:** read `~/AppStarter/starter.log`. To run the app directly and see
  the error:
  ```bash
  cd ~/AppStarter/releases/$(python3.8 -c "import json;print(json.load(open('$HOME/AppStarter/state.json'))['sha'])")
  ~/AppStarter/venv/bin/python main_1366.py
  ```
- **"Camera 0 could not be opened":** check `ls /dev/video*`, the `video` group (log out and in
  after setup), and that no other program holds the camera.
- **Packages failed to install:** start AppStarter again. It retries the install for a build
  whose packages are missing.
