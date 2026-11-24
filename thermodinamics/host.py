#!/usr/bin/env python3
"""Host for th.v on the FPGA (N and A are fixed by ./build.sh).
Shows the growth round by round, the four moves, and the order read from V1.

    python3 host.py [--port /dev/ttyUSB1]
"""
import argparse
import sys
import time

import serial

W, FRAC = 24, 16


def read_exact(ser, n):
    data = ser.read(n)
    if len(data) != n:
        sys.exit(f"timeout: got {len(data)} of {n} bytes (is the bitstream loaded? right port?)")
    return data


def snapshot(ser):
    ser.reset_input_buffer()
    ser.write(b"S")
    magic, nmax, n, a = read_exact(ser, 4)
    if magic != 0xA5:
        sys.exit(f"bad frame header 0x{magic:02x}")
    nbytes = (nmax * W + nmax + 1 + 7) // 8
    v = int.from_bytes(read_exact(ser, nbytes), "little")
    volts = []
    for x in range(nmax):
        raw = (v >> (1 + nmax + x * W)) & ((1 << W) - 1)
        if raw >> (W - 1):
            raw -= 1 << W
        volts.append(raw / (1 << FRAC))
    active = [(v >> (1 + x)) & 1 for x in range(nmax)]
    return dict(n=n, a=a, done=v & 1, active=active[:n], V=volts[:n])


def show(s, title):
    n, a = s["n"], s["a"]
    cells = [x for x in range(n) if s["active"][x]]
    v1 = s["V"][1]
    inv = f"{1 / v1:.2f}" if v1 > 0 else "inf"
    print(f"\n== {title}: active={len(cells)} done={s['done']} "
          f"charge={sum(s['V']):.5f}  1/V1={inv}")
    print("   " + "  ".join(f"V{x}={s['V'][x]:.4f}" for x in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()
    ser = serial.Serial(args.port, args.baud, timeout=2)

    ser.write(b"R")
    s = snapshot(ser)
    n, a = s["n"], s["a"]
    ainv = pow(a, -1, n)
    moves = [a, ainv, a * a % n, ainv * ainv % n]
    print(f"N={n} a={a}   moves: " + " ".join(f"x{m}" for m in moves))

    rnd = 0
    while True:                                   # growth, one ROUND+COMMIT at a time
        show(s, f"round {rnd}")
        if s["done"] or rnd > 64:
            break
        ser.write(b"T")
        rnd += 1
        s = snapshot(ser)

    ser.write(b"G")                               # free run until equilibrium
    time.sleep(0.3)
    s = snapshot(ser)
    show(s, "equilibrium")
    cells = [x for x in range(n) if s["active"][x]]
    print(f"\n   ord_{n}({a}) = {len(cells)} (active cells) = {1 / s['V'][1]:.2f} (1/V1)")
    print(f"   <{a}> = {cells}")

    print("\ncommands: t=round g=run p=pause s=snapshot d=text dump r=reset q=quit")
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "s":
            show(snapshot(ser), "snapshot")
        elif cmd == "d":
            ser.reset_input_buffer()
            ser.write(b"D")
            time.sleep(0.3)
            print(ser.read(ser.in_waiting).decode(errors="replace"), end="")
        elif cmd in ("t", "g", "p", "r"):
            ser.write(cmd.upper().encode())


if __name__ == "__main__":
    main()
