from sys import argv
import utils

def main(channel_name):
    utils.save_comment_threads_from_channel_uploads(channel_name)

if __name__ == '__main__':
    if len(argv) != 2:
        print('Usage: python3 driver.py <channel name>')
        exit(1)

    main(argv[1])
