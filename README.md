Welcome to my Applied Reinforcement Learning Term Project! To use this code, you will need the following hardware components:

  1. QTY:1 HANDi Hand including 6 Dynamixel XL330-M288-T motors
  2. QTY:2 TTL 3P Cable (JST - JST)
  3. QTY:1 Dynamixel U2D2 Power Hub Board (PHB)
  4. QTY:1 Dynamixel U2D2
  5. QTY:1 Female power jack
  6. QTY:1 Micro USB to USB cable
  7. QTY:1 5V 2A power supply

Follow these steps to connect your motor to your computer:
  1. Attach the female power jack to the green power connector on the U2D2 PHB.
  2. Attach the U2D2 to the PHB and connect them using one 3P cable using any 3-pin Dynamixel connector on the PHB and U2D2.
  3. Using the other 3P cable, connect the HANDi Hand to the U2D2 PHB. Plug the cable into any 3-pin Dynamixel connector on the PHB.
  4. Make sure the switch on the PHB is turned off (dotted side of switch is raised up).
  5. Use the Micro USB cable to connect the U2D2 to your computer.
  6. Use the 5V power supply to connect the PHB to a wall outlet.

To run the code:
  1. Go to device manager under Ports (COM & LPT) to find out which port your motor is connected to. Change the value in the variable "DEVICENAME" in the code to match.
  2. Switch on the PHB.
  3. Run the code from windows powershell by going into the folder where the code is and typing "python wiggle_rev5.py".
