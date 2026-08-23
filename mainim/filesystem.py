"""
Filesystem Animation using Manim
Animates the educational Unix-like filesystem described in the document.
Scenes:
  1. ArchitectureOverview   – layered architecture diagram
  2. PhysicalLayout         – disk.img block layout
  3. SuperblockScene        – superblock fields
  4. BitmapScene            – bitmap allocator in action
  5. InodeScene             – inode structure & block pointers
  6. DirectoryScene         – directory entries & path lookup
  7. WriteFileScene         – create_file + write_file walkthrough
  8. ReadFileScene          – read_file path-resolution chain
  9. DeleteScene            – delete & bitmap free
 10. FullTreeScene          – whole filesystem tree

Run a single scene:
  manim -pql filesystem_animation.py ArchitectureOverview

Run all scenes one after another (low quality for speed):
  manim -ql filesystem_animation.py  # renders each scene separately
"""

from manim import *

# ─── shared colour palette ────────────────────────────────────────────────────
C_SB = "#E63946"  # superblock  – red
C_BM = "#457B9D"  # bitmap      – steel-blue
C_IN = "#2A9D8F"  # inode       – teal
C_DIR = "#E9C46A"  # directory   – yellow
C_DATA = "#A8DADC"  # data block  – light-cyan
C_FREE = "#264653"  # free block  – dark
C_RES = "#6D6875"  # reserved    – purple-grey
C_TEXT = WHITE
C_ARROW = "#F4A261"  # arrow       – orange

BLOCK_W = 1.6
BLOCK_H = 0.55


def make_block(label, color, width=BLOCK_W, height=BLOCK_H, font_size=18):
    rect = Rectangle(
        width=width, height=height, color=color, fill_color=color, fill_opacity=0.25
    )
    text = Text(label, font_size=font_size, color=C_TEXT)
    text.scale_to_fit_width(width * 0.88)
    return VGroup(rect, text)


def make_labeled_block(block_num, label, color, width=BLOCK_W, height=BLOCK_H):
    g = make_block(label, color, width, height)
    num = Text(f"Block {block_num}", font_size=11, color=GRAY)
    num.next_to(g, UP, buff=0.06)
    return VGroup(g, num)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 1 – Architecture Overview
# ══════════════════════════════════════════════════════════════════════════════
class ArchitectureOverview(Scene):
    def construct(self):
        title = Text("Filesystem Architecture", font_size=36, color=C_ARROW)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        layers = [
            ("YOUR PROGRAM", "#FFFFFF", 0.4),
            ("FileSystem API", C_ARROW, 0.4),
            ("BlockDevice", C_SB, 0.4),
            ("disk.img  (file)", C_DATA, 0.4),
        ]

        boxes = VGroup()
        for label, col, _ in layers:
            b = RoundedRectangle(
                width=4.5,
                height=0.7,
                corner_radius=0.12,
                color=col,
                fill_color=col,
                fill_opacity=0.18,
            )
            t = Text(label, font_size=20, color=C_TEXT)
            t.move_to(b)
            boxes.add(VGroup(b, t))

        boxes.arrange(DOWN, buff=0.35)
        boxes.next_to(title, DOWN, buff=0.5)

        side_labels = ["Dirs  |  Inodes  |  BlockAllocator"]
        side = Text(side_labels[0], font_size=14, color=GRAY)
        side.next_to(boxes[1], RIGHT, buff=0.3)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_bottom(),
                boxes[i + 1].get_top(),
                buff=0.05,
                color=C_ARROW,
                stroke_width=2.5,
            )
            arrows.add(a)

        for b in boxes:
            self.play(FadeIn(b, shift=DOWN * 0.2), run_time=0.5)
        self.play(FadeIn(side))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.25))

        note = Text(
            "The FileSystem class coordinates\nall layers above disk.img",
            font_size=15,
            color=GRAY,
            line_spacing=1.2,
        )
        note.to_corner(DL, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 2 – Physical Layout of the Virtual Disk
# ══════════════════════════════════════════════════════════════════════════════
class PhysicalLayout(Scene):
    def construct(self):
        title = Text("Physical Disk Layout", font_size=34, color=C_ARROW)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        specs = [
            (0, "SUPERBLOCK", C_SB),
            (1, "BITMAP", C_BM),
            (2, "INODE TABLE", C_IN),
            ("3‑9", "RESERVED", C_RES),
            (10, "Root Dir", C_DIR),
            (11, "File Data", C_DATA),
            (12, "FREE", C_FREE),
            (13, "FREE", C_FREE),
        ]

        blocks = VGroup()
        for num, label, col in specs:
            b = make_labeled_block(num, label, col, width=3.2, height=0.58)
            blocks.add(b)

        blocks.arrange(DOWN, buff=0.12)
        blocks.next_to(title, DOWN, buff=0.4)
        blocks.scale(0.95)

        # left-side byte-offset annotations
        offsets = [
            "bytes 0–4095",
            "bytes 4096–8191",
            "bytes 8192–12287",
            "bytes 12288–40959",
            "bytes 40960–45055",
            "bytes 45056–49151",
            "",
            "",
        ]

        for i, (b, off) in enumerate(zip(blocks, offsets)):
            self.play(FadeIn(b, shift=RIGHT * 0.3), run_time=0.35)
            if off:
                ot = Text(off, font_size=11, color=GRAY)
                ot.next_to(b, RIGHT, buff=0.18)
                self.play(FadeIn(ot), run_time=0.25)

        caption = Text(
            "Every block = 4096 bytes\nblock_number × 4096 = byte offset",
            font_size=15,
            color=GRAY,
            line_spacing=1.3,
        )
        caption.to_corner(DL, buff=0.3)
        self.play(Write(caption))
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 3 – Superblock
# ══════════════════════════════════════════════════════════════════════════════
class SuperblockScene(Scene):
    def construct(self):
        title = Text("The Superblock  (Block 0)", font_size=34, color=C_SB)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        fields = [
            ("magic", "0xDEADBEEF", "filesystem identifier"),
            ("block_size", "4096", "bytes per block"),
            ("total_blocks", "N", "total blocks on disk"),
            ("inode_count", "1000", "max inodes"),
            ("free_blocks", "N - 11", "currently free blocks"),
            ("root_inode", "1", "inode number of  /"),
        ]

        table = VGroup()
        for field, value, desc in fields:
            row = VGroup(
                Text(field, font_size=17, color=C_SB),
                Text("=", font_size=17, color=GRAY),
                Text(value, font_size=17, color=C_ARROW),
                Text(f"  ← {desc}", font_size=14, color=GRAY),
            ).arrange(RIGHT, buff=0.18)
            table.add(row)

        table.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        table.next_to(title, DOWN, buff=0.5)
        table.to_edge(LEFT, buff=1.0)

        for row in table:
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.4)

        # highlight root_inode
        highlight = SurroundingRectangle(table[-1], color=C_ARROW, buff=0.08)
        note = Text(
            'root_inode = 1\n"Start at inode #1 for /"',
            font_size=16,
            color=C_ARROW,
            line_spacing=1.3,
        )
        note.next_to(highlight, RIGHT, buff=0.35)
        self.play(Create(highlight), Write(note))
        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 4 – Bitmap Allocator
# ══════════════════════════════════════════════════════════════════════════════
class BitmapScene(Scene):
    def construct(self):
        title = Text("Bitmap Allocator  (Block 1)", font_size=34, color=C_BM)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        n = 14  # blocks to show
        initial = [1] * 11 + [0] * 3  # first 11 allocated, rest free

        cells = VGroup()
        labels = VGroup()
        bits = VGroup()

        for i, bit in enumerate(initial):
            col = C_BM if bit else C_FREE
            sq = Square(side_length=0.55, color=col, fill_color=col, fill_opacity=0.35)
            bl = Text(str(bit), font_size=18, color=C_TEXT).move_to(sq)
            nl = Text(str(i), font_size=11, color=GRAY)
            nl.next_to(sq, DOWN, buff=0.06)
            cells.add(sq)
            bits.add(bl)
            labels.add(nl)

        group = VGroup(*[VGroup(c, b, l) for c, b, l in zip(cells, bits, labels)])
        group.arrange(RIGHT, buff=0.12)
        group.next_to(title, DOWN, buff=0.6)

        for g in group:
            self.play(FadeIn(g), run_time=0.1)

        # labels
        alloc_label = Text("ALLOCATED", font_size=14, color=C_BM)
        alloc_label.next_to(group[0], UP, buff=0.3)
        free_label = Text("FREE", font_size=14, color=C_FREE)
        free_label.next_to(group[11], UP, buff=0.3)
        self.play(Write(alloc_label), Write(free_label))

        # arrow to first free block
        arrow = Arrow(
            group[11].get_top() + UP * 0.6,
            group[11].get_top(),
            buff=0.05,
            color=C_ARROW,
            stroke_width=3,
        )
        arrow_label = Text("allocate_block() → 11", font_size=17, color=C_ARROW)
        arrow_label.next_to(arrow, UP, buff=0.1)
        self.play(GrowArrow(arrow), Write(arrow_label))
        self.wait(0.5)

        # flip bit 11 → 1
        new_sq = Square(
            side_length=0.55, color=C_BM, fill_color=C_BM, fill_opacity=0.35
        )
        new_sq.move_to(cells[11])
        new_bit = Text("1", font_size=18, color=C_TEXT).move_to(new_sq)
        self.play(
            Transform(cells[11], new_sq), Transform(bits[11], new_bit), run_time=0.7
        )

        result = Text("Block 11 now ALLOCATED", font_size=17, color=C_ARROW)
        result.next_to(group, DOWN, buff=0.45)
        self.play(Write(result))
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 5 – Inode Structure
# ══════════════════════════════════════════════════════════════════════════════
class InodeScene(Scene):
    def construct(self):
        title = Text("Inode Structure  (128 bytes each)", font_size=32, color=C_IN)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # inode box
        inode_fields = [
            "file_type",
            "size",
            "block_count",
            "direct_blocks[0..11]",
            "indirect_block",
            "double_indirect_block",
            "created / modified / accessed",
        ]

        inode_rects = VGroup()
        for f in inode_fields:
            r = Rectangle(
                width=3.4, height=0.46, color=C_IN, fill_color=C_IN, fill_opacity=0.18
            )
            t = Text(f, font_size=15, color=C_TEXT).move_to(r)
            inode_rects.add(VGroup(r, t))

        inode_rects.arrange(DOWN, buff=0.0)
        inode_rects.to_edge(LEFT, buff=1.0)
        inode_rects.shift(DOWN * 0.3)

        inode_title = Text("inode #2", font_size=18, color=C_IN)
        inode_title.next_to(inode_rects, UP, buff=0.1)
        self.play(FadeIn(inode_title))
        for r in inode_rects:
            self.play(FadeIn(r), run_time=0.25)

        # data blocks on the right
        data_blocks = VGroup()
        for i in range(3):
            b = make_block(f"Block #{11+i}", C_DATA, width=2.0, height=0.6)
            data_blocks.add(b)
        data_blocks.arrange(DOWN, buff=0.35)
        data_blocks.to_edge(RIGHT, buff=1.2)
        data_blocks.shift(DOWN * 0.5)

        self.play(FadeIn(data_blocks))

        # arrows from direct_blocks row → data blocks
        direct_row = inode_rects[3]
        colors_arr = [C_ARROW, YELLOW, GREEN]
        for i, (db, col) in enumerate(zip(data_blocks, colors_arr)):
            a = Arrow(
                direct_row.get_right(),
                db.get_left(),
                buff=0.1,
                color=col,
                stroke_width=2.5,
            )
            lbl = Text(f"direct[{i}]", font_size=12, color=col)
            lbl.next_to(a, UP, buff=0.05)
            self.play(GrowArrow(a), FadeIn(lbl), run_time=0.5)

        note = Text(
            "Indirect / double-indirect:\ndesigned but not yet implemented",
            font_size=14,
            color=GRAY,
            line_spacing=1.3,
        )
        note.to_corner(DR, buff=0.35)
        self.play(FadeIn(note))
        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 6 – Directory Entries & Path Lookup
# ══════════════════════════════════════════════════════════════════════════════
class DirectoryScene(Scene):
    def construct(self):
        title = Text("Directories  –  Path Lookup", font_size=32, color=C_DIR)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # DirEntry structure
        entry_parts = [
            ("inode\nnumber\n4 bytes", C_IN),
            ("name\nlength\n4 bytes", C_BM),
            ("name\n(UTF-8)\nvariable", C_DIR),
        ]
        entry_boxes = VGroup()
        for label, col in entry_parts:
            r = Rectangle(
                width=2.0, height=1.1, color=col, fill_color=col, fill_opacity=0.2
            )
            t = Text(label, font_size=14, color=C_TEXT, line_spacing=1.2).move_to(r)
            entry_boxes.add(VGroup(r, t))

        entry_boxes.arrange(RIGHT, buff=0)
        entry_boxes.shift(UP * 1.5)
        entry_title = Text("Directory Entry (DirEntry)", font_size=16, color=GRAY)
        entry_title.next_to(entry_boxes, UP, buff=0.12)
        self.play(Write(entry_title))
        for b in entry_boxes:
            self.play(FadeIn(b, shift=UP * 0.15), run_time=0.35)

        # concrete example
        example = VGroup(
            Text("02 00 00 00", font_size=15, color=C_IN),
            Text("|  09 00 00 00", font_size=15, color=C_BM),
            Text("|  hello.txt", font_size=15, color=C_DIR),
        ).arrange(RIGHT, buff=0.15)
        example.next_to(entry_boxes, DOWN, buff=0.3)
        self.play(FadeIn(example))
        self.wait(0.5)

        # lookup chain
        chain_items = [
            ("/hello.txt", C_TEXT),
            ("split →  parent=/  filename=hello.txt", GRAY),
            ("find inode(/)  →  inode #1", C_IN),
            ("lookup 'hello.txt'  in dir", C_DIR),
            ("→  inode #2", C_IN),
            ("→  direct_blocks[0] = 11", C_ARROW),
            ("→  Block #11", C_DATA),
            ('→  "Hello World"', GREEN),
        ]

        chain = VGroup()
        for label, col in chain_items:
            t = Text(label, font_size=16, color=col)
            chain.add(t)

        chain.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        chain.next_to(example, DOWN, buff=0.45)

        for t in chain:
            self.play(FadeIn(t, shift=RIGHT * 0.15), run_time=0.3)

        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 7 – create_file + write_file walkthrough
# ══════════════════════════════════════════════════════════════════════════════
class WriteFileScene(Scene):
    def construct(self):
        title = Text("create_file + write_file", font_size=32, color=C_ARROW)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── left column: step list ──────────────────────────────────────────
        steps = [
            ('fs.create_file("/hello.txt")', C_ARROW),
            ("split → parent=/ , name=hello.txt", GRAY),
            ("find inode #1  (root)", C_IN),
            ("allocate inode #2", C_IN),
            ("add  hello.txt→#2  to root dir", C_DIR),
            ('fs.write_file("/hello.txt", b"Hello World")', C_ARROW),
            ("data = 11 bytes → 1 block needed", GRAY),
            ("allocate block #11", C_BM),
            ('write "Hello World" → block #11', C_DATA),
            ("inode #2: size=11, direct[0]=11", C_IN),
        ]

        step_texts = VGroup()
        for s, col in steps:
            step_texts.add(Text(s, font_size=14, color=col))
        step_texts.arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        step_texts.to_edge(LEFT, buff=0.5)
        step_texts.shift(DOWN * 0.1)

        # ── right column: disk diagram ──────────────────────────────────────
        disk_blocks = VGroup(
            make_block("SB Block 0", C_SB, width=2.0, height=0.48),
            make_block("BM Block 1", C_BM, width=2.0, height=0.48),
            make_block("Inodes Block 2", C_IN, width=2.0, height=0.48),
            make_block("Reserved 3-9", C_RES, width=2.0, height=0.48),
            make_block("Root Dir Blk 10", C_DIR, width=2.0, height=0.48),
        )
        disk_blocks.arrange(DOWN, buff=0.08)
        disk_blocks.to_edge(RIGHT, buff=0.6)
        disk_blocks.shift(UP * 0.6)

        for b in disk_blocks:
            self.play(FadeIn(b), run_time=0.2)

        # animate each step
        markers = []
        highlights_done = set()

        for i, (txt, col) in enumerate(steps):
            self.play(FadeIn(step_texts[i], shift=RIGHT * 0.1), run_time=0.4)

            # add visual cue on the disk diagram for certain steps
            if i == 2:  # find inode #1
                hl = SurroundingRectangle(disk_blocks[2], color=C_IN, buff=0.05)
                self.play(Create(hl), run_time=0.3)
                markers.append(hl)

            elif i == 4:  # add to root dir
                hl = SurroundingRectangle(disk_blocks[4], color=C_DIR, buff=0.05)
                self.play(Create(hl), run_time=0.3)
                entry = Text("hello.txt→#2", font_size=11, color=C_DIR)
                entry.next_to(disk_blocks[4], RIGHT, buff=0.1)
                self.play(Write(entry), run_time=0.3)
                markers.append(hl)

            elif i == 8:  # write data block
                blk11 = make_block(
                    'Block 11\n"Hello World"', C_DATA, width=2.0, height=0.6
                )
                blk11.next_to(disk_blocks[4], DOWN, buff=0.08)
                self.play(FadeIn(blk11, shift=DOWN * 0.2))

            elif i == 9:  # update inode
                inode_note = Text("size=11  direct[0]=11", font_size=11, color=C_IN)
                inode_note.next_to(disk_blocks[2], RIGHT, buff=0.1)
                self.play(Write(inode_note), run_time=0.3)

        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 8 – Read File Path Resolution
# ══════════════════════════════════════════════════════════════════════════════
class ReadFileScene(Scene):
    def construct(self):
        title = Text('read_file("/hello.txt")', font_size=32, color=C_DATA)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        nodes = [
            ('"/hello.txt"', C_TEXT),
            ("_split_path()", GRAY),
            ('parent="/"  filename="hello.txt"', GRAY),
            ('_find_inode("/")', C_IN),
            ("inode #1", C_IN),
            ("_lookup_in_directory()", C_DIR),
            ('"hello.txt"', C_DIR),
            ("inode #2", C_IN),
            ("direct_blocks[0]", C_ARROW),
            ("block #11", C_DATA),
            ('"Hello World"', GREEN),
        ]

        boxes = VGroup()
        for label, col in nodes:
            r = RoundedRectangle(
                width=4.2,
                height=0.5,
                corner_radius=0.1,
                color=col,
                fill_color=col,
                fill_opacity=0.15,
            )
            t = Text(label, font_size=16, color=C_TEXT).move_to(r)
            boxes.add(VGroup(r, t))

        boxes.arrange(DOWN, buff=0.1)
        boxes.next_to(title, DOWN, buff=0.35)
        boxes.scale(0.92)

        arrows = VGroup()
        for i in range(len(boxes) - 1):
            a = Arrow(
                boxes[i].get_bottom(),
                boxes[i + 1].get_top(),
                buff=0.04,
                color=C_ARROW,
                stroke_width=2,
            )
            arrows.add(a)

        for i, (box, arr) in enumerate(zip(boxes, arrows)):
            self.play(FadeIn(box, shift=DOWN * 0.1), run_time=0.3)
            if i < len(arrows):
                self.play(GrowArrow(arr), run_time=0.2)

        self.play(FadeIn(boxes[-1], shift=DOWN * 0.1))

        # flash the final result
        self.play(Indicate(boxes[-1], color=GREEN, scale_factor=1.08))
        self.wait(2)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 9 – Delete Operation
# ══════════════════════════════════════════════════════════════════════════════
class DeleteScene(Scene):
    def construct(self):
        title = Text('delete("/docs/paper.txt")', font_size=32, color=C_SB)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        steps = [
            ("1. Find /docs", C_DIR),
            ("2. Find paper.txt in /docs", C_DIR),
            ("3. Get inode #4", C_IN),
            ("4. Free data blocks (bitmap 13→0)", C_BM),
            ("5. Delete inode #4", C_IN),
            ("6. Remove entry from /docs dir", C_DIR),
            ("7. _sync() metadata", GRAY),
        ]

        step_group = VGroup()
        for s, col in steps:
            t = Text(s, font_size=17, color=col)
            step_group.add(t)
        step_group.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        step_group.next_to(title, DOWN, buff=0.5)
        step_group.to_edge(LEFT, buff=1.2)

        # bitmap visualization
        n = 16
        bits = [1] * 14 + [0] * 2
        cells = VGroup()
        bit_texts = VGroup()
        for i, b in enumerate(bits):
            col = C_BM if b else C_FREE
            sq = Square(0.42, color=col, fill_color=col, fill_opacity=0.35)
            bt = Text(str(b), font_size=14, color=C_TEXT).move_to(sq)
            cells.add(sq)
            bit_texts.add(bt)

        bm_group = VGroup(*[VGroup(c, t) for c, t in zip(cells, bit_texts)])
        bm_group.arrange(RIGHT, buff=0.06)
        bm_group.to_edge(RIGHT, buff=0.5)
        bm_group.shift(UP * 0.5)
        bm_title = Text("Bitmap", font_size=16, color=C_BM)
        bm_title.next_to(bm_group, UP, buff=0.1)

        self.play(FadeIn(bm_title), FadeIn(bm_group))

        # animate steps
        for i, step in enumerate(step_group):
            self.play(FadeIn(step, shift=RIGHT * 0.15), run_time=0.45)

            if i == 3:  # free block 13 in bitmap
                free_sq = Square(
                    0.42, color=C_FREE, fill_color=C_FREE, fill_opacity=0.35
                )
                free_sq.move_to(cells[13])
                free_bt = Text("0", font_size=14, color=C_TEXT).move_to(free_sq)
                self.play(
                    Transform(cells[13], free_sq),
                    Transform(bit_texts[13], free_bt),
                    Flash(cells[13], color=C_ARROW, flash_radius=0.35),
                    run_time=0.7,
                )

        note = Text(
            "⚠  Old bytes may still exist in block 13\n"
            "   until something overwrites them.",
            font_size=14,
            color=GRAY,
            line_spacing=1.3,
        )
        note.to_corner(DR, buff=0.35)
        self.play(Write(note))
        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# Scene 10 – Full Filesystem Tree
# ══════════════════════════════════════════════════════════════════════════════
class FullTreeScene(Scene):
    def construct(self):
        title = Text("Full Filesystem Tree", font_size=34, color=C_ARROW)
        title.to_edge(UP, buff=0.25)
        self.play(Write(title))

        # ── tree nodes ──────────────────────────────────────────────────────
        def node(label, col, w=1.7, h=0.5):
            r = RoundedRectangle(
                width=w,
                height=h,
                corner_radius=0.1,
                color=col,
                fill_color=col,
                fill_opacity=0.22,
            )
            t = Text(label, font_size=14, color=C_TEXT).move_to(r)
            return VGroup(r, t)

        root = node("/  inode#1", C_DIR, w=1.6)
        hello = node("hello.txt #2", C_DATA)
        docs = node("docs/  #3", C_DIR)
        imgbin = node("image.bin #6", C_DATA)
        paper = node("paper.txt #4", C_DATA)
        notes = node("notes.txt #5", C_DATA)

        # positions
        root.move_to(UP * 2.5)
        hello.move_to(UP * 0.8 + LEFT * 3.5)
        docs.move_to(UP * 0.8)
        imgbin.move_to(UP * 0.8 + RIGHT * 3.5)
        paper.move_to(DOWN * 0.9 + LEFT * 1.0)
        notes.move_to(DOWN * 0.9 + RIGHT * 1.0)

        all_nodes = [root, hello, docs, imgbin, paper, notes]
        for n in all_nodes:
            self.play(FadeIn(n), run_time=0.3)

        # edges
        edges_def = [
            (root, hello),
            (root, docs),
            (root, imgbin),
            (docs, paper),
            (docs, notes),
        ]
        edge_colors = [C_DATA, C_DIR, C_DATA, C_DATA, C_DATA]

        for (a, b), col in zip(edges_def, edge_colors):
            line = Line(a.get_bottom(), b.get_top(), color=col, stroke_width=2)
            self.play(Create(line), run_time=0.3)

        # ── inode → block mapping table ────────────────────────────────────
        rows = [
            ("inode #2", "→ block #11", '"Hello..."', C_DATA),
            ("inode #3", "→ block #12", "dir entries", C_DIR),
            ("inode #4", "→ block #13", '"Paper..."', C_DATA),
            ("inode #5", "→ block #14", '"Notes..."', C_DATA),
            ("inode #6", "→ block #15", "image data", C_DATA),
        ]

        table = VGroup()
        for inode, arrow, data, col in rows:
            row = VGroup(
                Text(inode, font_size=13, color=C_IN),
                Text(arrow, font_size=13, color=C_ARROW),
                Text(data, font_size=13, color=col),
            ).arrange(RIGHT, buff=0.22)
            table.add(row)

        table.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        table.to_corner(DR, buff=0.5)
        table_title = Text("inode → block → data", font_size=14, color=GRAY)
        table_title.next_to(table, UP, buff=0.12)

        self.play(FadeIn(table_title))
        for row in table:
            self.play(FadeIn(row), run_time=0.25)

        self.wait(2.5)
