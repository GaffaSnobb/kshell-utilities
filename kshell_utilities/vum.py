import curses, time
from multiprocessing import managers
from typing import Callable

KEY_RETURN: int = 10
KEY_BACKSPACE: int = 127
KEY_TAB: int = 9

COLOR_DEFAULT_WHITE: int = 9
COLOR_DEFAULT_GREY: int = 10
COLOR_DEFAULT_RED: int = 11
COLOR_DEFAULT_GREEN: int = 12
COLOR_DEFAULT_DARK_GREY: int = 13
COLOR_DIM_WHITE: int = 14
COLOR_BLACK: int = 15

COLOR_DEFAULT_PAIR: int = 1
COLOR_RED_PAIR: int = 2
COLOR_GREEN_PAIR: int = 3
COLOR_DARK_GREY_PAIR: int = 4
COLOR_DEBUG_PAIR: int = 5
COLOR_DIM_PAIR: int = 6
COLOR_SUBHEADER: int = 7

class Vum:
    def __init__(self,
        shared_dict: managers.DictProxy = {},
        screen = None,
        command_prompt_icon: str = ">",
        action_interval: int | float = 0.5,
        is_command_log_enabled: bool = True,
    ) -> None:
        """
        Vum is the Vi-unimproved. It is a simple text-based user
        interface for the kshell utilities. It is a wrapper for the
        curses library.

        Parameters
        ----------
        shared_dict : managers.DictProxy
            A dictionary that is shared between processes. This is
            used to communicate between the main process and the
            subprocesses that are spawned for the user interface if
            such parallelisation is required. NOTE: Currently not
            properly implemented.

        screen :
            The curses screen object. May be provided by the main
            process if needed. Defaults to None which in turn creates
            a new initscr.

        command_prompt_icon : str
            The icon that is displayed before the command prompt.

        action_interval : int | float
            The curses getkey function is non-blocking and a custom
            user provided function is called at regular intervals.

        is_command_log_enabled : bool
            If True, the command log is displayed in the bottom left
            corner of the screen.
        """
        if screen is None:
            screen = curses.initscr()
        
        self.shared_dict = shared_dict
        self.screen = screen
        self.command_prompt_icon = f" {command_prompt_icon} "
        self.action_interval = action_interval
        self.is_command_log_enabled = is_command_log_enabled

        self.n_rows, self.n_cols = self.screen.getmaxyx()
        self.time_fmt: str = '%H:%M:%S'  # Formatter for strftime.

        self.command_log_length: int = 5
        self.command_log = [""]*self.command_log_length

        screen.keypad(True)
        screen.nodelay(True)
        curses.echo(False)

        curses.start_color()
        curses.init_color(COLOR_DEFAULT_WHITE, 1000, 1000, 1000)  # Default font white.
        curses.init_color(COLOR_DEFAULT_GREY, 118, 118, 118)    # Default background grey.
        curses.init_color(COLOR_DEFAULT_RED, 1000, 0, 0)    # Default font red.
        curses.init_color(COLOR_DEFAULT_GREEN, 0, 1000, 0)    # Default font green.
        curses.init_color(COLOR_DEFAULT_DARK_GREY, 80, 80, 80)    # Default background grey.
        curses.init_color(COLOR_DIM_WHITE, 500, 500, 500)  # Dim white for blinking cursor.
        curses.init_color(COLOR_BLACK, 0, 0, 0)  # Dim white for blinking cursor.
        # curses.init_pair(COLOR_DEFAULT_PAIR, COLOR_DEFAULT_WHITE, COLOR_DEFAULT_GREY)    # Default font-background color pair.
        curses.init_pair(COLOR_DEFAULT_PAIR, COLOR_DEFAULT_WHITE, COLOR_BLACK)    # Default font-background color pair.
        curses.init_pair(COLOR_RED_PAIR, COLOR_DEFAULT_RED, COLOR_DEFAULT_DARK_GREY)
        curses.init_pair(COLOR_GREEN_PAIR, COLOR_DEFAULT_GREEN, COLOR_DEFAULT_DARK_GREY)
        curses.init_pair(COLOR_DARK_GREY_PAIR, COLOR_DEFAULT_WHITE, COLOR_DEFAULT_DARK_GREY)
        curses.init_pair(COLOR_DEBUG_PAIR, COLOR_DEFAULT_WHITE, COLOR_DEFAULT_RED)
        curses.init_pair(COLOR_DIM_PAIR, COLOR_DIM_WHITE, COLOR_BLACK)    # Default font-background color pair.
        curses.init_pair(COLOR_SUBHEADER, COLOR_DIM_WHITE, COLOR_DEFAULT_DARK_GREY)    # Default font-background color pair.

        self.screen.bkgd(' ', curses.color_pair(COLOR_DEFAULT_PAIR))
        # self.screen.bkgd(' ', curses.color_pair(COLOR_DARK_GREY_PAIR))

    def addstr(self,
        y: int = 0,
        x: int = 0,
        string: str = "",
        is_blank_line: bool = True,
    ) -> None:
        """
        Wrapper for blanking a line, adding a new string to the same
        line and then refresh the screen.

        Parameters
        ----------
        y : int
            The y coordinate of the string.
        
        x : int
            The x coordinate of the string.
        
        string : str
            The string to be added to the screen.

        is_blank_line : bool
            If True, the line is first blanked before adding the
            string.
        """
        if is_blank_line: self.screen.addstr(y, x, self.blank_line)
        self.screen.addstr(y, x, string)
        self.screen.refresh()

    def input(self,
        command_prompt_message: str,
        action: Callable = lambda: None,
    ) -> str:
        """
        Prompt for user input. Return the user input as a string when
        the return key is pressed.

        Parameters
        ----------
        command_prompt_message : str
            The message to be displayed before the command prompt.

        action : Callable
            A function that is called at regular intervals.

        Returns
        -------
        str
            The user input after the return key is pressed.
        """
        x_offset: int = len(command_prompt_message) + len(self.command_prompt_icon) # x coord. for the command text field.
        cursor_pos: list[int] = [self.n_rows - 1, x_offset]
        msg: str = ""
        self.blank_line: str = " "*(self.n_cols - 1)
        refresh_timer: float = 0.0

        current_attr = COLOR_DIM_PAIR   # For the blinking command prompt icon.
        previous_attr = COLOR_DEFAULT_PAIR

        max_msg_length: int = self.n_cols - max( # Make sure that user input length never exceeds the width of the text prompt.
            (len(command_prompt_message) + len(self.command_prompt_icon)),
            len(time.strftime(self.time_fmt, time.localtime())) + len(self.command_prompt_icon)
        )
        while True:
            time.sleep(0.01)  # To reduce the CPU load.
            self.screen.addstr(self.n_rows - 1, 0, command_prompt_message)

            if abs(time.perf_counter() - refresh_timer) > self.action_interval:
                """
                For adding some action to be performed every
                `action_interval` seconds. For example, making the
                command prompt icon blink (super important).
                """
                refresh_timer = time.perf_counter()
                action()
                self.screen.addstr(self.n_rows - 1, len(command_prompt_message), self.command_prompt_icon, curses.color_pair(current_attr))
                current_attr, previous_attr = previous_attr, current_attr
                self.screen.refresh()

            self.screen.move(*cursor_pos)
            try:
                """
                When nodelay is set to True, then getkey raises an error
                when no input is detected.
                """
                input_ = self.screen.getkey()
            except curses.error:
                continue

            self.screen.refresh()

            if input_ == "KEY_LEFT":
                cursor_pos[1] -= 1 if (cursor_pos[1] > x_offset) else 0
                continue

            if input_ == "KEY_RIGHT":
                cursor_pos[1] += 1 if ((cursor_pos[1] < (self.n_cols - 1)) and cursor_pos[1] < len(msg) + x_offset) else 0
                continue
        
            if input_ == "KEY_UP":
                msg = self.command_log[-1].split(self.command_prompt_icon)[-1].strip()
                self.screen.addstr(self.n_rows - 1, x_offset, msg)
                cursor_pos[1] = len(msg) + x_offset
                continue

            if input_ == "KEY_DC":
                self.screen.addstr(self.n_rows - 1, x_offset, self.blank_line[x_offset:])
                cursor_pos[1] -= 1 if (cursor_pos[1] == (len(msg) + 1) + x_offset) else 0
                msg = msg[:cursor_pos[1] - x_offset] + msg[cursor_pos[1] - x_offset + 1:] if (cursor_pos[1] < (len(msg) + 1) + x_offset) else msg
                self.screen.addstr(self.n_rows - 1, x_offset, msg)
                continue

            if len(input_) > 1:
                """
                'Keypad keys' are strings of more than 1 character, while
                any other input is just a single character. Skip any
                unhandled keypad keys.
                """
                continue

            if ord(input_) == KEY_BACKSPACE:
                if cursor_pos[1] == x_offset: continue
                self.screen.addstr(self.n_rows - 1, x_offset, self.blank_line[x_offset:])
                cursor_pos[1] -= 1 if (cursor_pos[1] > x_offset) else 0
                msg = msg[:cursor_pos[1] - x_offset] + msg[cursor_pos[1] - x_offset + 1:]
                self.screen.addstr(self.n_rows - 1, x_offset, msg)
                continue

            if ord(input_) == KEY_RETURN:
                """
                Handle commands.
                """
                if not msg: continue    # Empty string is not a command and should not clutter the log.
                # self.screen.addstr(self.n_rows - 1, x_offset, self.blank_line[x_offset:])
                self.screen.addstr(self.n_rows - 1, 0, self.blank_line)
                command: str = msg
                msg = ""
                self.command_log.pop(0)
                self.command_log.append(f"{time.strftime(self.time_fmt, time.localtime())}{self.command_prompt_icon}{command}")
                # self._logger(command)
                cursor_pos[1] = x_offset
                if self.is_command_log_enabled:
                    self.logger()
                return command
            
            # if ord(input_) == KEY_TAB:
            #     if not msg: continue    # Skip tab completion for empty string.

            #     msg, matches = tab_completer(msg=msg, allowed_values=current_auto_complete_options)

            #     self.command_window.addstr(self.n_rows - 1, x_offset, blank_line[x_offset:])
            #     self.command_window.addstr(self.n_rows - 1, x_offset, msg)
            #     cursor_pos[1] = len(msg) + x_offset
            #     continue

            if len(msg) >= (max_msg_length - 1): continue
            msg = msg[:cursor_pos[1] - x_offset] + input_ + msg[cursor_pos[1] - x_offset:]
            cursor_pos[1] += 1
            self.screen.addstr(self.n_rows - 1, x_offset, msg)

    def logger(self) -> None:
        """
        Display the command log.
        """
        for i in range(self.command_log_length):
            self.screen.addstr(self.n_rows - self.command_log_length + i - 1, 0, self.blank_line)
            self.screen.addstr(self.n_rows - self.command_log_length + i - 1, 0, self.command_log[i])

        self.screen.refresh()