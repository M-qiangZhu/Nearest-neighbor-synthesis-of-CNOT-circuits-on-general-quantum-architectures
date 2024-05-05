// Initial wiring: [0, 1, 4, 5, 6, 15, 9, 10, 13, 3, 11, 2, 7, 8, 12, 14]
// Resulting wiring: [0, 1, 4, 5, 6, 15, 9, 10, 13, 3, 11, 2, 7, 8, 12, 14]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[7], q[6];
cx q[8], q[7];
cx q[7], q[6];
cx q[13], q[10];
cx q[10], q[9];
cx q[9], q[8];
cx q[10], q[9];
cx q[15], q[14];
cx q[14], q[15];
cx q[10], q[11];
cx q[11], q[12];
cx q[9], q[10];
cx q[9], q[14];
cx q[10], q[11];
cx q[14], q[15];
cx q[11], q[12];
cx q[6], q[9];
cx q[9], q[14];
cx q[6], q[7];
cx q[5], q[6];
cx q[2], q[3];
cx q[0], q[1];
