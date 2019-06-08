import re
import string
import sys
from collections import Counter


def is_valid(line):
    l = len(line)
    if l > 1000000 or l < 50:
        return False
    count = Counter(line)
    alpha_cnt = sum(count[ch] for ch in string.ascii_letters)
    if alpha_cnt < 50 or alpha_cnt / l < 0.7:
        return False
    if count['/'] / l > 0.05:  # filter hyperlinks
        return False
    if count['\\'] / l > 0.05:  # filter latex math equations
        return False
    if count['|'] / l > 0.05 or line[0] == '|':  # filter remaining tables
        return False
    return True


def post_cleanup(line):
    line = re.sub(r'\\', ' ', line)  # remove all backslashes
    return ' '.join(line.strip().split())  # remove redundant spaces


existed = set()
pending_tail = string.ascii_letters + string.digits + ','


def write_output(line):
    global existed
    if is_valid(line):
        line = post_cleanup(line)
        if line not in existed:
            existed.add(line)
            sys.stdout.write(line + '\n')


def check_concat(line1, line2):
    global pending_tail
    if len(line1) == 0 or len(line2) == 0:
        return False
    return (line1[-1] in pending_tail) and (line2[0] in string.ascii_lowercase)


def main():
    buf = []
    for line in sys.stdin:
        line = ' '.join(line.strip().split())
        if buf and (not check_concat(buf[-1], line)):
            write_output(' '.join(buf) + '\n')
            buf = []
        buf.append(line)
    if buf:
        write_output(' '.join(buf) + '\n')


if __name__ == '__main__':
    main()
