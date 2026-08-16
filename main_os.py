import os

def run_scripts():

    for i in range(6):
        print(f"开始训练第{i}次...")
        #os.system(f"python main_train.py --num {i}")
        #os.system(f"python compare/main_asyffnet.py --num {i}")
        #os.system(f"python compare/main_gsanet.py --num {i}")
        os.system(f"python compare/main_dfinet.py --num {i}")
        # os.system(f"python compare/main_nuo.py --num {i}")

if __name__ == "__main__":
    run_scripts()
