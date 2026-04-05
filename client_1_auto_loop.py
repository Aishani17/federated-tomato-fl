import os
import sys
import time
import subprocess
import glob

CLIENT_ID = "Client_1"

BASE_DIR = os.getcwd()
SIGNALS_DIR = os.path.join(BASE_DIR, "Signals")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "client_1_train_pi.py")

PYTHON = sys.executable

os.makedirs(SIGNALS_DIR, exist_ok=True)

print(f"🔄 {CLIENT_ID} FL Node Started.")
print(f"📡 Monitoring for Server Signals in: {SIGNALS_DIR}")
print(f"🐍 Using Python: {PYTHON}")

def clear_old_signals():
    for file in glob.glob(os.path.join(SIGNALS_DIR, "*.txt")):
        os.remove(file)

def get_start_rounds():
    return glob.glob(os.path.join(SIGNALS_DIR, "start_round_*.txt"))

def main():
    clear_old_signals()

    while True:
        signals = get_start_rounds()

        if signals:
            signals.sort()
            current_signal = signals[0]

            round_str = os.path.basename(current_signal).replace("start_round_", "").replace(".txt", "")

            print("\n==========================================")
            print(f"📥 Received start_round_{round_str}.txt")
            print(f"🚀 Starting Round {round_str}")
            print("==========================================")

            try:
                subprocess.run(
                    [PYTHON, "-u", TRAIN_SCRIPT, "--config", current_signal],
                    check=True
                )

                os.remove(current_signal)

                done_file = os.path.join(SIGNALS_DIR, f"done_round_{round_str}.txt")
                with open(done_file, "w") as f:
                    f.write("done")

                print(f"📡 Sent done_round_{round_str}.txt")

            except Exception as e:
                print("❌ Error:", e)

        time.sleep(5)

if __name__ == "__main__":
    main()