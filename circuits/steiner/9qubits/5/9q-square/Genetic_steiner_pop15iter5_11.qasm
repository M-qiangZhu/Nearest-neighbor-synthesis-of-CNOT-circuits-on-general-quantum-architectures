// Initial wiring: [6, 7, 1, 2, 8, 3, 5, 4, 0]
// Resulting wiring: [6, 7, 1, 2, 8, 3, 5, 4, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[1], q[4];
cx q[0], q[1];
cx q[1], q[4];
cx q[6], q[7];
cx q[4], q[1];
cx q[1], q[0];
