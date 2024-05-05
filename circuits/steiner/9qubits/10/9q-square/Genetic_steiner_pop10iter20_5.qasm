// Initial wiring: [6, 1, 0, 8, 3, 5, 7, 2, 4]
// Resulting wiring: [6, 1, 0, 8, 3, 5, 7, 2, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[1], q[4];
cx q[5], q[6];
cx q[4], q[5];
cx q[5], q[4];
cx q[4], q[3];
cx q[5], q[4];
cx q[4], q[5];
cx q[4], q[1];
cx q[1], q[0];
cx q[4], q[1];
cx q[3], q[4];
cx q[0], q[1];
