// Initial wiring: [8, 3, 16, 7, 19, 1, 9, 12, 13, 2, 10, 15, 11, 5, 14, 18, 4, 6, 17, 0]
// Resulting wiring: [8, 3, 16, 7, 19, 1, 9, 12, 13, 2, 10, 15, 11, 5, 14, 18, 4, 6, 17, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[8], q[2];
cx q[16], q[15];
cx q[12], q[0];
cx q[17], q[2];
cx q[14], q[5];
cx q[19], q[3];
cx q[14], q[19];
cx q[9], q[18];
cx q[7], q[18];
cx q[6], q[9];
cx q[6], q[19];
cx q[10], q[17];
cx q[7], q[15];
cx q[2], q[4];
cx q[4], q[19];
cx q[2], q[18];
