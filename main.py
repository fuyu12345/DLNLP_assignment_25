import subprocess

def main():
    try:
        # Execute the shell script
        print("Starting the training process using run.sh...")
        # replace with the  shell script to run the training and evaluation
        subprocess.run(["bash", "run.sh"], check=True) 
        print("successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training: {e}")
    except FileNotFoundError:
        print("run.sh not found. Please make sure it is in the same directory as main.py or provide the correct path.")

if __name__ == "__main__":
    main()
