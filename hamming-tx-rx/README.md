# Hamming-TXRX
Implementation in Simulink of a Hamming code Transmitter-Receiver system, with code correction

This simulink project simulate a TX-RX digital system, where an error on one bit is simulated on the channel.

In addition to the payload, also 3 bit are transmitted generated using the hamming code.
In the receiver, those bits are used to detect and correct the error that happened on the channel.

You can select on which bit the error should occour. 
The Hamming code is an auto-correction code, which let you to automatically correct an error happened on the channel.

In the project I've also added a powerpoint file that briefly explain the functioning of the system.
