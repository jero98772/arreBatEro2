"""
Render with (Manim Community Edition):
    manim -pql round_robin.py ProcessAutomaton
    manim -pql round_robin.py RoundRobinMultiCPU

Two scenes:
  1) ProcessAutomaton     -> finite-state automaton of a process:
                             NEW -> READY -> RUNNING -> BLOCKED -> EXIT
  2) RoundRobinMultiCPU   -> Round Robin scheduling across several CPUs,
                             processes visibly moving between
                             NEW / CPU(RUNNING) / READY / BLOCKED / EXIT lanes.
"""

import numpy as np
from manim import *

config.background_color = "#0e1117"

STATE_COLORS = {
    "NEW": GRAY_B,
    "READY": BLUE,
    "RUNNING": GREEN,
    "BLOCKED": RED,
    "EXIT": GRAY_D,
}


def clipped_arc(
    center_a,
    center_b,
    angle,
    radius_a=0.9,
    radius_b=0.9,
    color=WHITE,
    stroke_width=3,
    tip_length=0.2,
    samples=200,
):
    """An ArcBetweenPoints whose endpoints are trimmed so the curve (and its
    arrow tip) stop at the circle boundaries instead of piercing into the
    circles / their text labels."""
    full = ArcBetweenPoints(
        center_a, center_b, angle=angle, color=color, stroke_width=stroke_width
    )
    ts = np.linspace(0, 1, samples)
    pts = [full.point_from_proportion(t) for t in ts]

    t_start = 0.0
    for t, p in zip(ts, pts):
        if np.linalg.norm(p - center_a) >= radius_a:
            t_start = t
            break
    t_end = 1.0
    for t, p in zip(reversed(ts), reversed(pts)):
        if np.linalg.norm(p - center_b) >= radius_b:
            t_end = t
            break

    trimmed = full.copy()
    trimmed.pointwise_become_partial(full, t_start, t_end)
    trimmed.add_tip(tip_length=tip_length)
    return trimmed


# ============================================================
# SCENE 1 : PROCESS STATE AUTOMATON
# ============================================================
class ProcessAutomaton(Scene):
    def construct(self):
        title = Text("Process State Automaton", font_size=34).to_edge(UP, buff=0.4)
        self.play(Write(title))

        pos = {
            "NEW": LEFT * 5 + UP * 0.6,
            "READY": LEFT * 1.3 + UP * 0.6,
            "RUNNING": RIGHT * 2.3 + UP * 0.6,
            "BLOCKED": RIGHT * 2.3 + DOWN * 2.4,
            "EXIT": RIGHT * 5.7 + UP * 0.6,
        }
        radius = 0.85

        states = {}
        for name, p in pos.items():
            circle = Circle(radius=radius, color=STATE_COLORS[name], fill_opacity=0.25)
            circle.move_to(p)
            label = Text(name, font_size=20).move_to(p)
            states[name] = VGroup(circle, label)

        self.play(*[FadeIn(g) for g in states.values()])
        self.wait(0.3)

        def straight_edge(a, b, text, label_dir=UP, buff=0.15):
            start = states[a][0].get_center()
            end = states[b][0].get_center()
            arrow = Arrow(
                start,
                end,
                buff=radius + 0.05,
                color=WHITE,
                stroke_width=3,
                max_tip_length_to_length_ratio=0.08,
            )
            label = Text(text, font_size=15, color=YELLOW)
            label.next_to(arrow.get_center(), label_dir, buff=buff)
            return VGroup(arrow, label)

        def curved_edge(a, b, text, angle, label_dir, buff=0.2):
            start = states[a][0].get_center()
            end = states[b][0].get_center()
            arrow = clipped_arc(
                start, end, angle=angle, radius_a=radius, radius_b=radius
            )
            mid = arrow.point_from_proportion(0.5)
            label = Text(text, font_size=15, color=YELLOW)
            label.next_to(mid, label_dir, buff=buff)
            return VGroup(arrow, label)

        e_admit = straight_edge("NEW", "READY", "admit")
        e_dispatch = straight_edge("READY", "RUNNING", "dispatch")
        e_timeout = curved_edge(
            "RUNNING", "READY", "timeout / preempt", angle=1.1, label_dir=UP, buff=0.25
        )
        e_wait = straight_edge(
            "RUNNING", "BLOCKED", "I/O or event wait", label_dir=RIGHT, buff=0.2
        )
        e_wakeup = curved_edge(
            "BLOCKED",
            "READY",
            "I/O or event complete",
            angle=1.1,
            label_dir=DOWN,
            buff=0.3,
        )
        e_exit = straight_edge("RUNNING", "EXIT", "exit")

        for e in [e_admit, e_dispatch, e_timeout, e_wait, e_wakeup, e_exit]:
            self.play(Create(e[0]), FadeIn(e[1]), run_time=0.6)

        self.wait(0.4)

        # Animate a token traveling through a sample lifecycle.
        # It rests just BELOW each circle so it never overlaps the state text.
        def marker_pos(name):
            return states[name][0].get_center() + DOWN * (radius + 0.3)

        token = Dot(color=YELLOW, radius=0.14).move_to(marker_pos("NEW"))
        self.play(FadeIn(token))

        path_nodes = ["NEW", "READY", "RUNNING", "BLOCKED", "READY", "RUNNING", "EXIT"]
        for nxt in path_nodes[1:]:
            self.play(token.animate.move_to(marker_pos(nxt)), run_time=0.8)
            self.play(
                Flash(states[nxt][0], color=STATE_COLORS[nxt], flash_radius=1.0),
                run_time=0.3,
            )

        self.wait(1)


# ============================================================
# SCENE 2 : ROUND ROBIN SCHEDULING, MULTIPLE PROCESSORS
# ============================================================
NUM_CPUS = 3
QUANTUM = 1  # one "tick" per round (each tick = 1 quantum)
IO_BLOCK_DURATION = 2  # rounds a process stays BLOCKED after I/O

PROCESSES = {
    0: {"burst": 6, "io_at": 3, "color": "#e74c3c"},
    1: {"burst": 4, "io_at": None, "color": "#3498db"},
    2: {"burst": 8, "io_at": 4, "color": "#2ecc71"},
    3: {"burst": 3, "io_at": None, "color": "#f1c40f"},
    4: {"burst": 5, "io_at": 2, "color": "#9b59b6"},
    5: {"burst": 7, "io_at": None, "color": "#1abc9c"},
}

# Row centers, spaced to leave clear room for the title above and a margin
# below (frame half-height is 4.0 at default aspect ratio).
Y_NEW, Y_CPU, Y_READY, Y_BLOCKED, Y_EXIT = 2.5, 1.15, -0.2, -1.55, -2.9
ROW_BOX_H = 0.95
LANE_START_X, LANE_STEP = -4.4, 1.8  # first process slot in a row, and spacing
CPU_STEP = 3.2
LABEL_X = -5.7  # row-name caption, safely inside the box


def lane_slot(index, y):
    return np.array([LANE_START_X + index * LANE_STEP, y, 0])


def cpu_slot(index):
    total_width = (NUM_CPUS - 1) * CPU_STEP
    start = -total_width / 2
    return np.array([start + index * CPU_STEP, Y_CPU, 0])


class RoundRobinMultiCPU(Scene):
    def construct(self):
        title = Text(
            "Round Robin Scheduling — Multiple Processors", font_size=26
        ).to_edge(UP, buff=0.35)
        self.play(Write(title))

        # ---------- lane backgrounds + short row labels ----------
        lanes = [
            ("NEW", Y_NEW),
            ("CPUs", Y_CPU),
            ("READY", Y_READY),
            ("BLOCKED", Y_BLOCKED),
            ("EXIT", Y_EXIT),
        ]
        for name, y in lanes:
            box = RoundedRectangle(
                width=12.6,
                height=ROW_BOX_H,
                corner_radius=0.15,
                color=GRAY_D,
                fill_opacity=0.08,
                stroke_opacity=0.6,
            )
            box.move_to([0, y, 0])
            lbl = Text(name, font_size=15, color=GRAY_B).move_to([LABEL_X, y, 0])
            self.add(box, lbl)

        # CPU index tags sit in the gap between the NEW row and the CPU row.
        gap_y = (Y_NEW - ROW_BOX_H / 2 + Y_CPU + ROW_BOX_H / 2) / 2
        for i in range(NUM_CPUS):
            cpu_lbl = Text(f"CPU {i}", font_size=13, color=GREEN)
            cpu_lbl.move_to([cpu_slot(i)[0], gap_y, 0])
            self.add(cpu_lbl)

        round_counter = Text("Round: 0", font_size=20, color=YELLOW)
        round_counter.move_to([5.6, title.get_center()[1], 0])
        self.add(round_counter)

        # ---------- build process mobjects ----------
        proc_mobs = {}
        initial_order = list(PROCESSES.keys())
        for i, pid in enumerate(initial_order):
            color = PROCESSES[pid]["color"]
            circle = Circle(
                radius=0.38, color=color, fill_color=color, fill_opacity=0.85
            )
            id_text = Text(f"P{pid}", font_size=15, color=WHITE)
            burst_text = Text(str(PROCESSES[pid]["burst"]), font_size=13, color=WHITE)
            burst_text.next_to(circle, DOWN, buff=0.08)
            group = VGroup(circle, id_text, burst_text)
            group.move_to(lane_slot(i, Y_NEW))
            proc_mobs[pid] = group
            self.add(group)

        self.wait(0.3)

        # NEW -> READY (admit all processes)
        self.play(
            *[
                proc_mobs[pid].animate.move_to(lane_slot(i, Y_READY))
                for i, pid in enumerate(initial_order)
            ],
            run_time=1.0,
        )

        # ---------- scheduling state ----------
        ready_order = list(initial_order)
        cpu_order = [None] * NUM_CPUS
        blocked_order = []
        exit_order = []
        remaining = {pid: PROCESSES[pid]["burst"] for pid in PROCESSES}
        consumed = {pid: 0 for pid in PROCESSES}
        io_done = {pid: False for pid in PROCESSES}
        blocked_timer = {}

        def refresh_positions(run_time=0.6):
            anims = []
            for idx, pid in enumerate(ready_order):
                anims.append(proc_mobs[pid].animate.move_to(lane_slot(idx, Y_READY)))
            for idx, pid in enumerate(blocked_order):
                anims.append(proc_mobs[pid].animate.move_to(lane_slot(idx, Y_BLOCKED)))
            for idx, pid in enumerate(exit_order):
                anims.append(proc_mobs[pid].animate.move_to(lane_slot(idx, Y_EXIT)))
            for i, pid in enumerate(cpu_order):
                if pid is not None:
                    anims.append(proc_mobs[pid].animate.move_to(cpu_slot(i)))
            if anims:
                self.play(*anims, run_time=run_time)

        round_num = 0
        MAX_ROUNDS = 60
        while len(exit_order) < len(PROCESSES) and round_num < MAX_ROUNDS:
            round_num += 1
            new_counter = Text(f"Round: {round_num}", font_size=20, color=YELLOW)
            new_counter.move_to(round_counter.get_center())
            self.play(Transform(round_counter, new_counter), run_time=0.3)

            # 1) assign idle CPUs from the ready queue
            for i in range(NUM_CPUS):
                if cpu_order[i] is None and ready_order:
                    cpu_order[i] = ready_order.pop(0)
            refresh_positions(run_time=0.6)

            running_pids = [pid for pid in cpu_order if pid is not None]
            if not running_pids:
                continue

            # 2) run one quantum tick: pulse + update remaining-burst labels
            pulse_anims = [
                Indicate(proc_mobs[pid][0], color=YELLOW, scale_factor=1.2)
                for pid in running_pids
            ]
            label_anims = []
            for pid in running_pids:
                remaining[pid] -= QUANTUM
                consumed[pid] += QUANTUM
                old_label = proc_mobs[pid][2]
                new_label = Text(str(max(remaining[pid], 0)), font_size=13, color=WHITE)
                new_label.move_to(old_label.get_center())
                label_anims.append(Transform(old_label, new_label))
            self.play(*pulse_anims, *label_anims, run_time=0.6)

            # 3) decide outcome for each running process
            finished_now, blocked_now, requeued_now = [], [], []
            for i, pid in enumerate(cpu_order):
                if pid is None:
                    continue
                if remaining[pid] <= 0:
                    finished_now.append(pid)
                    cpu_order[i] = None
                elif (
                    PROCESSES[pid]["io_at"] is not None
                    and not io_done[pid]
                    and consumed[pid] >= PROCESSES[pid]["io_at"]
                ):
                    io_done[pid] = True
                    blocked_timer[pid] = IO_BLOCK_DURATION
                    blocked_now.append(pid)
                    cpu_order[i] = None
                else:
                    requeued_now.append(pid)
                    cpu_order[i] = None

            for pid in finished_now:
                exit_order.append(pid)
            for pid in blocked_now:
                blocked_order.append(pid)
            for pid in requeued_now:
                ready_order.append(pid)

            refresh_positions(run_time=0.7)

            # dim finished processes
            if finished_now:
                self.play(
                    *[
                        proc_mobs[pid][0]
                        .animate.set_fill(GRAY, opacity=0.4)
                        .set_stroke(GRAY)
                        for pid in finished_now
                    ],
                    run_time=0.4,
                )

            # 4) tick down blocked processes, wake up any that are ready
            woken_now = []
            for pid in list(blocked_timer.keys()):
                blocked_timer[pid] -= 1
                if blocked_timer[pid] <= 0:
                    woken_now.append(pid)
                    del blocked_timer[pid]

            for pid in woken_now:
                blocked_order.remove(pid)
                ready_order.append(pid)

            if woken_now:
                refresh_positions(run_time=0.6)

        self.wait(1)
        summary = Text("All processes completed", font_size=26, color=GREEN)
        summary.move_to(title.get_center())
        self.play(Transform(title, summary), FadeOut(round_counter))
        self.wait(2)
