// Initial wiring: [7, 5, 11, 12, 8, 4, 10, 14, 15, 0, 1, 3, 6, 2, 9, 13]
// Resulting wiring: [7, 5, 11, 12, 8, 4, 10, 14, 15, 0, 1, 3, 6, 2, 9, 13]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[2], q[1];
cx q[4], q[3];
cx q[6], q[5];
cx q[7], q[6];
cx q[6], q[1];
cx q[7], q[6];
cx q[9], q[8];
cx q[8], q[7];
cx q[9], q[8];
cx q[11], q[4];
cx q[14], q[13];
cx q[14], q[9];
cx q[9], q[14];
cx q[9], q[10];
cx q[14], q[9];
cx q[8], q[15];
cx q[6], q[9];
cx q[9], q[14];
cx q[9], q[10];
cx q[14], q[9];
cx q[5], q[10];
