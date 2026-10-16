"""Drive the optical reader of a Molecular Devices FlexStation II without SoftMax Pro.

The FlexStation II reader shares its optical design with the SpectraMax Gemini
family, which speaks an ASCII "!COMMAND" protocol over RS-232. PyLabRobot ships
a reverse-engineered driver for the Gemini EM. Whether the FlexStation II
firmware accepts the same commands is not documented; run `probe` first.

Subcommands
  probe  Read-only handshake at several baud rates. Prints raw replies.
  send   Send one command (e.g. "!OPEN") and print the raw reply.
  read   Endpoint fluorescence read of a full 96-well plate via PyLabRobot.

Requirements
  pip install pyserial
  pip install "pylabrobot @ git+https://github.com/PyLabRobot/pylabrobot.git"  # for `read`
"""

import argparse
import asyncio
import csv
import sys
import time

BAUD_RATES = [9600, 19200, 38400, 57600, 115200]

# Query-only commands: none of these moves the carriage or the pipettor.
PROBE_COMMANDS = ["!OPTION", "!STATUS", "!ERROR", "!TEMP"]


def _exchange(ser, command: str, wait: float = 1.5) -> bytes:
  ser.reset_input_buffer()
  ser.write(command.encode() + b"\r")
  deadline = time.time() + wait
  reply = b""
  while time.time() < deadline:
    chunk = ser.read(ser.in_waiting or 1)
    if chunk:
      reply += chunk
      # Gemini-family firmware ends each response field with '>'.
      if reply.rstrip().endswith(b">") and "OK" in reply.decode(errors="replace"):
        break
  return reply


def probe(port: str, bauds) -> int:
  import serial

  for baud in bauds:
    print(f"--- {port} @ {baud} baud")
    try:
      with serial.Serial(port, baudrate=baud, bytesize=8, parity="N", stopbits=1, timeout=0.2) as ser:
        hello = _exchange(ser, "!")
        print(f"  '!'  -> {hello!r}")
        if not hello:
          continue
        for cmd in PROBE_COMMANDS:
          print(f"  {cmd:8s} -> {_exchange(ser, cmd)!r}")
        print(f"\nInstrument answered at {baud} baud.")
        return 0
    except serial.SerialException as exc:
      print(f"  could not open port: {exc}")
      return 2
  print("\nNo reply at any baud rate. Check cable (null-modem vs straight), port name, and power.")
  return 1


def send(port: str, baud: int, command: str) -> int:
  import serial

  with serial.Serial(port, baudrate=baud, timeout=0.2) as ser:
    print(repr(_exchange(ser, command, wait=30)))
  return 0


def _load_backend():
  try:
    from pylabrobot.legacy.plate_reading.molecular_devices import (
      MolecularDevicesSpectraMaxGeminiEMBackend as Backend,
    )
  except ImportError:
    from pylabrobot.plate_reading.molecular_devices import (  # older releases
      MolecularDevicesSpectraMaxGeminiEMBackend as Backend,
    )
  return Backend


async def read(args) -> int:
  from pylabrobot.resources.corning.plates import Cor_96_wellplate_360ul_Fb

  backend = _load_backend()(port=args.port)
  if args.baud != 9600:
    backend.io.baudrate = args.baud  # PyLabRobot defaults to 9600
  plate = Cor_96_wellplate_360ul_Fb(name="plate")

  await backend.setup()
  try:
    results = await backend.read_fluorescence(
      plate=plate,
      excitation_wavelength=args.ex,
      emission_wavelength=args.em,
      read_from_bottom=args.bottom,
      flashes_per_well=args.flashes,
    )
  finally:
    await backend.stop()

  rows = "ABCDEFGH"
  with open(args.out, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["ex_nm", "em_nm", "temperature_c", "well", "rfu"])
    for read_block in results:
      grid = read_block["data"]  # rows x columns
      for r, row in enumerate(grid):
        for c, value in enumerate(row):
          writer.writerow([read_block.get("ex_wavelength"), read_block.get("em_wavelength"),
                           read_block.get("temperature"), f"{rows[r]}{c + 1}", value])
  print(f"Wrote {args.out}")
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="cmd", required=True)

  p = sub.add_parser("probe")
  p.add_argument("--port", required=True, help="e.g. COM3 or /dev/ttyUSB0")
  p.add_argument("--baud", type=int, help="test only this baud rate")

  s = sub.add_parser("send")
  s.add_argument("--port", required=True)
  s.add_argument("--baud", type=int, default=9600)
  s.add_argument("command", help='e.g. "!OPEN"')

  r = sub.add_parser("read")
  r.add_argument("--port", required=True)
  r.add_argument("--baud", type=int, default=9600)
  r.add_argument("--ex", type=int, required=True, help="excitation wavelength, nm")
  r.add_argument("--em", type=int, required=True, help="emission wavelength, nm")
  r.add_argument("--bottom", action="store_true", help="bottom read (cell assays)")
  r.add_argument("--flashes", type=int, default=6)
  r.add_argument("--out", default="flexstation_read.csv")

  args = parser.parse_args()
  if args.cmd == "probe":
    return probe(args.port, [args.baud] if args.baud else BAUD_RATES)
  if args.cmd == "send":
    return send(args.port, args.baud, args.command)
  return asyncio.run(read(args))


if __name__ == "__main__":
  sys.exit(main())
