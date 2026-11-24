// ============================================================================
//  diffusion_computer.v   (UN SOLO ARCHIVO)
//
//  Contiene:
//    diffusion_computer : el nucleo (plano digital + difusion en punto fijo)
//    uart_rx / uart_tx  : puerto serie
//    fpga_top           : modulo superior para la FPGA (nucleo + UART)
//    tb, tb_uart        : testbenches (solo con -DSIMULATION)
//
//  ---------------------------- FPGA ----------------------------------------
//   Modulo superior (top) : fpga_top
//   Parametros a ajustar  : N, A, CLK_HZ (frecuencia del reloj de tu placa)
//   Pines                 : clk, uart_rxd, uart_txd, led_done, led_run
//   (los tb quedan fuera de la sintesis porque estan bajo `ifdef SIMULATION)
//
//  ---------------------------- PROTOCOLO UART (115200 8N1) ------------------
//   PC -> FPGA, un byte ASCII por comando:
//     'R'  reinicia (solo C1 activa, V1 = 1) y pausa
//     'T'  avanza UN paso (un ROUND+COMMIT del crecimiento y de la difusion)
//     'G'  corre libre (un paso por cada ciclo de reloj)
//     'P'  pausa
//     'S'  pausa y envia una foto del estado
//   FPGA -> PC, respuesta a 'S':
//     0xA5, luego NBYTES bytes = vector  { v_flat , active , done }  en
//     little-endian (byte 0 = bits 7..0). Es decir:
//        bit 0               done (el crecimiento ya se saturo)
//        bits 1 .. N         active[0 .. N-1]
//        bits N+1 ...        V[x] en bits [N+1 + x*W +: W], con signo, 1.0 = 2^FRAC
//     NBYTES = ceil((N*W + N + 1) / 8)
//
//  ---------------------------- SIMULACION -----------------------------------
//   Nucleo (N y A configurables):
//     iverilog -g2005 -DSIMULATION -s tb -P tb.N=21 -P tb.A=2 -o sim diffusion_computer.v
//     vvp sim
//   Nucleo + UART (simula los comandos y decodifica la respuesta):
//     iverilog -g2005 -DSIMULATION -s tb_uart -o sim_uart diffusion_computer.v
//     vvp sim_uart
// ============================================================================
`timescale 1ns/1ps

module diffusion_computer #(
    parameter N     = 21,   // módulo
    parameter A     = 2,    // elemento generador
    parameter FRAC  = 16,   // bits fraccionarios: 1.0 = 2^FRAC
    parameter W     = 24,   // ancho del voltaje (con signo)
    parameter SHIFT = 3     // alpha = 1/2^SHIFT
)(
    input  wire           clk,
    input  wire           rst,
    input  wire           ce,         // clock enable: 1 = avanza un paso (ROUND+COMMIT)
    output wire [N-1:0]   active_o,   // bits ACTIVE de todas las celdas
    output wire           done_o,     // 1 cuando el crecimiento ya se saturó
    output wire [31:0]    order_o,    // popcount(active) = orden de A (cuando done)
    output wire [N*W-1:0] v_flat      // voltajes, celda x en v_flat[x*W +: W]
);

    // ------------------------------------------------------------------
    // Aritmética modular en tiempo de elaboración
    // ------------------------------------------------------------------
    function integer modinv;
        input integer a_in;
        input integer m;
        integer k;
        begin
            modinv = 0;                       // 0 = "no existe inverso"
            for (k = 1; k < m; k = k + 1)
                if (((a_in * k) % m) == 1) modinv = k;
        end
    endfunction

    localparam AM    = A % N;
    localparam AINV  = modinv(AM, N);         // 2^-1  mod 21 = 11
    localparam A2    = (AM   * AM)   % N;     // 2^2   mod 21 = 4
    localparam A2INV = (AINV * AINV) % N;     // 2^-2  mod 21 = 16
    localparam signed [W-1:0] ONE = 1 <<< FRAC;

`ifdef SIMULATION
    initial begin
        if (AINV == 0) begin
            $display("ERROR: A=%0d no es unidad modulo N=%0d (gcd(A,N) != 1)", A, N);
            $finish;
        end
    end
`endif

    genvar x;

    // ------------------------------------------------------------------
    // PLANO DIGITAL: crecimiento simultáneo de estrellas
    //
    // La celda x recibe 4 líneas (una por red R_g) y las combina con un OR.
    // Como cada R_g es una permutación, "quién me envía a mí" es x*g^-1:
    //    R_A    : origen = x*A^-1      R_Ainv : origen = x*A
    //    R_A2   : origen = x*A^-2      R_A2inv: origen = x*A^2
    // Todo son constantes de elaboración => es puro cableado fijo.
    // ------------------------------------------------------------------
    reg  [N-1:0] active;
    wire [N-1:0] next;

    generate
        for (x = 0; x < N; x = x + 1) begin : G_NEXT
            assign next[x] = active[(AINV  * x) % N]    // llega por R_A
                           | active[(AM    * x) % N]    // llega por R_Ainv
                           | active[(A2INV * x) % N]    // llega por R_A2
                           | active[(A2    * x) % N];   // llega por R_A2inv
        end
    endgenerate

    // COMMIT: ACTIVE <- ACTIVE | NEXT.   Al inicio solo C1 está activa.
    always @(posedge clk) begin
        if (rst) active <= ({{(N-1){1'b0}}, 1'b1} << 1);
        else if (ce) active <= active | next;
    end

    // El crecimiento terminó cuando NEXT no aporta nada nuevo
    assign done_o   = ((next & ~active) == {N{1'b0}});
    assign active_o = active;

    // popcount(active) = |<A>| una vez que done_o = 1
    integer p;
    reg [31:0] cnt;
    always @* begin
        cnt = 0;
        for (p = 0; p < N; p = p + 1) cnt = cnt + active[p];
    end
    assign order_o = cnt;

    // ------------------------------------------------------------------
    // PLANO ANALÓGICO (simulado en punto fijo): difusión por aristas
    //
    // Cada arista no dirigida es {x, A*x} o {x, A^2*x}; las otras dos
    // direcciones (A^-1, A^-2) son esas mismas aristas vistas desde el otro
    // extremo. Una arista existe solo si sus DOS extremos están activos.
    // ------------------------------------------------------------------
    reg  signed [W-1:0] V   [0:N-1];
    wire signed [W-1:0] fA  [0:N-1];   // flujo x -> A*x
    wire signed [W-1:0] fA2 [0:N-1];   // flujo x -> A^2*x

    // d / 2^SHIFT redondeando hacia cero (evita sesgo en el redondeo)
    function signed [W-1:0] div_pow2_trunc;
        input signed [W-1:0] d;
        begin
            if (d < 0) div_pow2_trunc = -((-d) >>> SHIFT);
            else       div_pow2_trunc =   d    >>> SHIFT;
        end
    endfunction

    generate
        for (x = 0; x < N; x = x + 1) begin : G_FLUX
            localparam TA  = (AM * x) % N;
            localparam TA2 = (A2 * x) % N;
            assign fA[x]  = (active[x] & active[TA])
                            ? div_pow2_trunc(V[x] - V[TA])  : 0;
            assign fA2[x] = (active[x] & active[TA2])
                            ? div_pow2_trunc(V[x] - V[TA2]) : 0;
        end
    endgenerate

    // Cada celda: pierde lo que envía, gana lo que le envían
    integer k;
    always @(posedge clk) begin
        if (rst) begin
            for (k = 0; k < N; k = k + 1) V[k] <= 0;
            V[1] <= ONE;                              // V1(0) = 1, resto 0
        end else if (ce) begin
            for (k = 0; k < N; k = k + 1)
                V[k] <= V[k] - fA[k] - fA2[k]
                             + fA [(AINV  * k) % N]   // me envía x = k*A^-1
                             + fA2[(A2INV * k) % N];  // me envía x = k*A^-2
        end
    end

    // Voltajes aplanados para sacarlos por el puerto
    generate
        for (x = 0; x < N; x = x + 1) begin : G_OUT
            assign v_flat[x*W +: W] = V[x];
        end
    endgenerate

endmodule

// ============================================================================
//  UART receptor  (8N1, muestrea a mitad de cada bit)
// ============================================================================
module uart_rx #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115200
)(
    input  wire       clk,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid          // pulso de 1 ciclo cuando llega un byte
);
    localparam CPB = CLK_HZ / BAUD;  // ciclos de reloj por bit

    reg        rx1, rx2, busy;
    reg [15:0] cnt;
    reg [3:0]  bitn;
    initial begin rx1 = 1; rx2 = 1; busy = 0; cnt = 0; bitn = 0; data = 0; valid = 0; end

    always @(posedge clk) begin
        rx1 <= rx;  rx2 <= rx1;      // sincroniza la entrada asincrona
        valid <= 1'b0;
        if (!busy) begin
            if (!rx2) begin          // flanco de bajada = bit de inicio
                busy <= 1; cnt <= CPB/2 - 1; bitn <= 0;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1;
        end else begin
            cnt <= CPB - 1;
            if (bitn == 0) begin
                if (rx2) busy <= 0;                  // falso arranque
                else     bitn <= 1;
            end else if (bitn <= 8) begin
                data <= {rx2, data[7:1]};            // LSB primero
                bitn <= bitn + 1;
            end else begin
                busy  <= 0;
                valid <= rx2;                        // bit de parada correcto
            end
        end
    end
endmodule

// ============================================================================
//  UART transmisor
// ============================================================================
module uart_tx #(
    parameter CLK_HZ = 100_000_000,
    parameter BAUD   = 115200
)(
    input  wire       clk,
    input  wire [7:0] data,
    input  wire       start,         // pulso de 1 ciclo (solo si busy = 0)
    output reg        busy,
    output reg        tx
);
    localparam CPB = CLK_HZ / BAUD;

    reg [15:0] cnt;
    reg [3:0]  bitn;
    reg [9:0]  sh;                   // {parada, dato[7:0], inicio}
    initial begin busy = 0; tx = 1; cnt = 0; bitn = 0; sh = 10'h3FF; end

    always @(posedge clk) begin
        if (!busy) begin
            tx <= 1'b1;
            if (start) begin
                sh <= {1'b1, data, 1'b0};
                busy <= 1; cnt <= CPB - 1; bitn <= 0;
                tx <= 1'b0;                          // bit de inicio
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1;
        end else begin
            cnt <= CPB - 1;
            if (bitn == 9) busy <= 0;
            else begin
                tx   <= sh[bitn + 1];
                bitn <= bitn + 1;
            end
        end
    end
endmodule

// ============================================================================
//  fpga_top : nucleo + UART + LEDs.  ESTE es el modulo superior para la FPGA.
// ============================================================================
module fpga_top #(
    parameter N      = 21,
    parameter A      = 2,
    parameter CLK_HZ = 100_000_000,   // <-- frecuencia del reloj de tu placa
    parameter BAUD   = 115200
)(
    input  wire clk,
    input  wire uart_rxd,             // pin RX de la FPGA (viene del PC)
    output wire uart_txd,             // pin TX de la FPGA (va al PC)
    output wire led_done,             // 1 = crecimiento terminado
    output wire led_run               // 1 = corriendo libre
);
    localparam FRAC   = 16;
    localparam W      = 24;
    localparam SHIFT  = 3;
    localparam NB     = N*W + N + 1;
    localparam NBYTES = (NB + 7) / 8;

    // reset de encendido (~16 ciclos); los registros arrancan en 0 en la FPGA
    reg [3:0] por;
    initial por = 4'd0;
    wire por_busy = ~&por;
    always @(posedge clk) if (por_busy) por <= por + 4'd1;

    reg cmd_rst, run, step, sending, tx_start;
    reg [7:0]  tx_data;
    reg [15:0] idx;
    initial begin cmd_rst = 0; run = 0; step = 0; sending = 0; tx_start = 0; tx_data = 0; idx = 0; end

    wire core_rst = por_busy | cmd_rst;
    wire ce       = run | step;

    wire [N-1:0]   active;
    wire           done;
    wire [31:0]    order;
    wire [N*W-1:0] v_flat;

    diffusion_computer #(.N(N), .A(A), .FRAC(FRAC), .W(W), .SHIFT(SHIFT)) core (
        .clk(clk), .rst(core_rst), .ce(ce),
        .active_o(active), .done_o(done), .order_o(order), .v_flat(v_flat)
    );

    wire [7:0] rx_data;
    wire       rx_valid;
    wire       tx_busy;

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rx(uart_rxd), .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .data(tx_data), .start(tx_start), .busy(tx_busy), .tx(uart_txd));

    // vector que se envia en la foto
    wire [NBYTES*8-1:0] vec;
    assign vec = {v_flat, active, done};

    always @(posedge clk) begin
        cmd_rst  <= 1'b0;
        step     <= 1'b0;
        tx_start <= 1'b0;

        // ---- comandos ----
        if (rx_valid && !sending) begin
            case (rx_data)
                "R": begin cmd_rst <= 1'b1; run <= 1'b0; end
                "T": step <= 1'b1;
                "G": run  <= 1'b1;
                "P": run  <= 1'b0;
                "S": begin run <= 1'b0; sending <= 1'b1; idx <= 16'd0; end
                default: ;
            endcase
        end

        // ---- envio de la foto: cabecera 0xA5 y luego NBYTES bytes ----
        if (sending && !tx_busy && !tx_start) begin
            if (idx == 16'd0) tx_data <= 8'hA5;
            else              tx_data <= vec[8*(idx-1) +: 8];
            tx_start <= 1'b1;
            if (idx == NBYTES) sending <= 1'b0;
            else               idx     <= idx + 16'd1;
        end
    end

    assign led_done = done;
    assign led_run  = run;
endmodule

`ifdef SIMULATION

module tb;
    parameter N     = 21;
    parameter A     = 2;
    parameter FRAC  = 16;
    parameter W     = 24;
    parameter SHIFT = 3;
    parameter NCYC  = 300;   // ciclos de simulación (crece ~ orden^2)

    reg clk;
    reg rst;
    wire [N-1:0]   active;
    wire           done;
    wire [31:0]    order;
    wire [N*W-1:0] v_flat;

    diffusion_computer #(.N(N), .A(A), .FRAC(FRAC), .W(W), .SHIFT(SHIFT)) dut (
        .clk(clk), .rst(rst), .ce(1'b1),
        .active_o(active), .done_o(done), .order_o(order), .v_flat(v_flat)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // --- orden multiplicativo de A mod N, calculado "a mano" (referencia) ---
    integer ref_order, t, it;
    initial begin
        ref_order = 0;
        t = A % N;
        for (it = 1; it <= N; it = it + 1) begin
            if (t == 1 && ref_order == 0) ref_order = it;
            t = (t * A) % N;
        end
    end

    // --- utilidades ---
    integer x, total, cyc, done_cycle, errors, tol;
    real    vreal;
    reg signed [W-1:0] vx;

    task read_total;
        begin
            total = 0;
            for (x = 0; x < N; x = x + 1) begin
                vx = v_flat[x*W +: W];
                total = total + vx;
            end
        end
    endtask

    initial begin
        rst = 1; cyc = 0; done_cycle = -1; errors = 0;
        repeat (2) @(negedge clk);
        rst = 0;

        $display("N=%0d  A=%0d  (orden esperado = %0d)", N, A, ref_order);
        $display("ciclo | #activas | done |   V1      | carga_total");

        for (cyc = 1; cyc <= NCYC; cyc = cyc + 1) begin
            @(negedge clk);
            read_total;

            vx = v_flat[1*W +: W];
            vreal = $itor(vx) / $itor(1 << FRAC);

            if (done && done_cycle < 0) done_cycle = cyc - 1;

            if (cyc <= 8 || cyc % 50 == 0)
                $display("%5d | %8d |  %0d   | %8.5f  | %0d", cyc, order, done, vreal, total);

            if (total !== (1 << FRAC)) begin
                errors = errors + 1;
                if (errors < 5) $display("  !! carga NO conservada en ciclo %0d: %0d", cyc, total);
            end
        end

        // --- verificaciones finales ---
        $display("----------------------------------------------");
        $display("crecimiento terminado tras %0d rondas COMMIT", done_cycle + 1);

        if (order !== ref_order) begin
            $display("FALLO: order_o=%0d, esperado=%0d", order, ref_order);
            errors = errors + 1;
        end else
            $display("OK: order_o = %0d = ord_%0d(%0d)", order, N, A);

        tol = 8 * ref_order;                       // tolerancia: zona muerta del redondeo
        $display("equilibrio esperado: %0d LSB por celda (1/%0d), tolerancia +-%0d",
                 (1 << FRAC) / ref_order, ref_order, tol);
        for (x = 0; x < N; x = x + 1) begin
            vx = v_flat[x*W +: W];
            if (active[x]) begin
                if ((vx - (1 << FRAC) / ref_order > tol) ||
                    ((1 << FRAC) / ref_order - vx > tol)) begin
                    $display("  celda %0d fuera de tolerancia: %0d", x, vx);
                    errors = errors + 1;
                end
            end else if (vx !== 0) begin
                $display("  celda inactiva %0d tiene carga %0d", x, vx);
                errors = errors + 1;
            end
        end

        if (errors == 0) $display("RESULTADO: PASS");
        else             $display("RESULTADO: FAIL (%0d errores)", errors);
        $finish;
    end

    // Para GTKWave
    initial begin
        $dumpfile("diffusion.vcd");
        $dumpvars(0, tb);
    end
endmodule


// ============================================================================
//  tb_uart : simula un PC que manda comandos por UART y decodifica la foto
// ============================================================================
module tb_uart;
    localparam N      = 21;
    localparam A      = 2;
    localparam W      = 24;
    localparam FRAC   = 16;
    localparam CLK_HZ = 10_000_000;        // reloj lento para simular rapido
    localparam BAUD   = 115200;
    localparam CPB    = CLK_HZ / BAUD;
    localparam NB     = N*W + N + 1;
    localparam NBYTES = (NB + 7) / 8;

    reg  clk, rxd;
    wire txd, led_done, led_run;
    initial begin clk = 0; rxd = 1; end
    always #50 clk = ~clk;                 // 10 MHz

    fpga_top #(.N(N), .A(A), .CLK_HZ(CLK_HZ), .BAUD(BAUD)) dut (
        .clk(clk), .uart_rxd(rxd), .uart_txd(txd),
        .led_done(led_done), .led_run(led_run));

    // ---- PC -> FPGA ----
    task send_byte;
        input [7:0] b;
        integer i;
        begin
            rxd = 0;                       repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin
                rxd = b[i];                repeat (CPB) @(posedge clk);
            end
            rxd = 1;                       repeat (2*CPB) @(posedge clk);
        end
    endtask

    // ---- FPGA -> PC (receptor del testbench) ----
    reg [7:0] rxbuf [0:NBYTES+8];
    integer   rxn, j;
    reg [7:0] rb;
    initial rxn = 0;
    always begin
        @(negedge txd);
        repeat (CPB/2) @(posedge clk);
        repeat (CPB)   @(posedge clk);
        for (j = 0; j < 8; j = j + 1) begin
            rb[j] = txd;                   repeat (CPB) @(posedge clk);
        end
        rxbuf[rxn] = rb;
        rxn = rxn + 1;
    end

    // ---- pide una foto y la decodifica ----
    reg [NBYTES*8-1:0] vec;
    reg signed [W-1:0] vx;
    integer x, i, t, nact, total, errors, tol;
    reg     done_f;

    task snapshot;
        begin
            rxn = 0;
            send_byte("S");
            wait (rxn == NBYTES + 1);
            repeat (2*CPB) @(posedge clk);
            if (rxbuf[0] !== 8'hA5) begin
                $display("  !! cabecera incorrecta: %h", rxbuf[0]);
                errors = errors + 1;
            end
            for (i = 0; i < NBYTES; i = i + 1) vec[8*i +: 8] = rxbuf[1+i];
            done_f = vec[0];
            nact = 0; total = 0;
            for (x = 0; x < N; x = x + 1) begin
                vx = vec[N+1 + x*W +: W];
                total = total + vx;
                if (vec[1+x]) nact = nact + 1;
            end
        end
    endtask

    initial begin
        errors = 0;
        repeat (50) @(posedge clk);

        $display("--- paso a paso: R, luego T + S ---");
        send_byte("R");
        for (t = 1; t <= 4; t = t + 1) begin
            send_byte("T");
            snapshot;
            vx = vec[N+1 + 1*W +: W];
            $display("paso %0d: activas=%0d done=%0d V1=%8.5f carga=%0d",
                     t, nact, done_f, $itor(vx) / $itor(1 << FRAC), total);
            if (total !== (1 << FRAC)) errors = errors + 1;
            if (t == 2 && (nact !== 6 || done_f !== 1'b1 || vx !== (1 << (FRAC-1)))) begin
                $display("  !! esperaba 6 activas, done=1 y V1=0.5 tras 2 pasos");
                errors = errors + 1;
            end
        end
        $display("FRAME %h", vec);          // para probar el decodificador de Python

        $display("--- corrida libre: G, espera, S ---");
        send_byte("G");
        repeat (200) @(posedge clk);
        snapshot;
        tol = 8 * 6;
        $display("equilibrio: activas=%0d done=%0d carga=%0d (esperado cada una ~%0d)",
                 nact, done_f, total, (1 << FRAC) / 6);
        if (nact !== 6 || total !== (1 << FRAC)) errors = errors + 1;
        for (x = 0; x < N; x = x + 1) begin
            vx = vec[N+1 + x*W +: W];
            if (vec[1+x]) begin
                if ((vx - (1 << FRAC)/6 > tol) || ((1 << FRAC)/6 - vx > tol)) begin
                    $display("  celda %0d fuera de tolerancia: %0d", x, vx);
                    errors = errors + 1;
                end
            end
        end

        if (errors == 0) $display("RESULTADO UART: PASS");
        else             $display("RESULTADO UART: FAIL (%0d errores)", errors);
        $finish;
    end
endmodule
`endif