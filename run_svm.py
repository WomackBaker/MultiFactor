import argparse
import subprocess
import sys
import os

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {command}", file=sys.stderr)
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_users', type=int, default=100)
    parser.add_argument('--gen_rows', type=int, default=1000)
    parser.add_argument('--data_samples', type=int, default=10000)
    parser.add_argument('--attackers', type=int, default=2000)
    args = parser.parse_args()

    python_cmd = 'python.exe' if os.name == 'nt' else 'python'

    run_command(f"cd GenerateData && {python_cmd} generate.py {args.gen_users} {args.gen_rows}")
    run_command(f"cd GAN && {python_cmd} gan.py {args.data_samples}")
    run_command(f"cd SVM && {python_cmd} split.py {args.attackers}")
    run_command(f"cd SVM && {python_cmd} svm.py")
    run_command(f"cd SVM && {python_cmd} fpr_fnr.py")

if __name__ == '__main__':
    main()
