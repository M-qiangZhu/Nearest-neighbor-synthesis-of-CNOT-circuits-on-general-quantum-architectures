// Initial wiring: [5 1 2 3 6 0 7 4 8]
// Resulting wiring: [5 0 2 4 6 1 7 3 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[5];
cx q[4], q[1];
cx q[0], q[1];
cx q[0], q[1];
cx q[0], q[1];
cx q[3], q[4];
cx q[3], q[4];
cx q[3], q[4];
cx q[7], q[4];
cx q[2], q[1];
cx q[0], q[1];
cx q[5], q[4];
cx q[6], q[7];
cx q[7], q[6];
