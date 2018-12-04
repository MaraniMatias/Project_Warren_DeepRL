import sys


# Show a progress bar
def updateProgress(progress, tick="", total="", status="Loading..."):
    line_length = 80
    bar_length = 23
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Waiting...\r"
    if progress >= 1:
        progress = 1
        status = "Completed loading data\r\n"
    block = int(round(bar_length * progress))
    line = str("\rImage: {0}/{1} [{2}] {3}% {4}").format(
        tick,
        total,
        str(("#" * block)) + str("." * (bar_length - block)),
        round(progress * 100, 1),
        status,
    )
    empty_block = line_length - len(line)
    empty_block = " " * empty_block if empty_block > 0 else ""
    sys.stdout.write(line + empty_block)
    sys.stdout.flush()
    if progress == 1:
        print("")
