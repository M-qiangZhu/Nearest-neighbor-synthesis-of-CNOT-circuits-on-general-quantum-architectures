// Initial wiring: [5, 3, 4, 6, 2, 1, 8, 7, 0]
// Resulting wiring: [5, 3, 4, 6, 2, 1, 8, 7, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[1], q[4];
cx q[4], q[5];
cx q[4], q[7];
cx q[3], q[4];
cx q[7], q[6];
cx q[6], q[5];
cx q[7], q[6];
cx q[6], q[7];
cx q[3], q[2];
cx q[1], q[0];
