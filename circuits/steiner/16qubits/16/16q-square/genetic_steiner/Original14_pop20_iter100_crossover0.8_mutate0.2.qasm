// Initial wiring: [8, 1, 11, 9, 6, 0, 15, 10, 14, 13, 12, 7, 5, 3, 2, 4]
// Resulting wiring: [8, 1, 11, 9, 6, 0, 15, 10, 14, 13, 12, 7, 5, 3, 2, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[4], q[3];
cx q[7], q[6];
cx q[6], q[5];
cx q[10], q[9];
cx q[9], q[6];
cx q[10], q[5];
cx q[13], q[10];
cx q[10], q[9];
cx q[9], q[8];
cx q[10], q[9];
cx q[13], q[10];
cx q[14], q[13];
cx q[15], q[14];
cx q[15], q[8];
cx q[13], q[14];
cx q[12], q[13];
cx q[13], q[14];
cx q[5], q[10];
cx q[5], q[6];
cx q[2], q[5];
cx q[5], q[10];
cx q[5], q[6];
