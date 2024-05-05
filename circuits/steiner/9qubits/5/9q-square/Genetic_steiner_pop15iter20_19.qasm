// Initial wiring: [1, 7, 3, 0, 6, 4, 5, 2, 8]
// Resulting wiring: [1, 7, 3, 0, 6, 4, 5, 2, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[5];
cx q[6], q[5];
cx q[5], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[4], q[3];
cx q[5], q[4];
cx q[2], q[1];
