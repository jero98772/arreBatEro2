

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
    parameter N              = 21,
    parameter A              = 2,
    parameter CLK_HZ         = 27_000_000,   // Tang Nano 20K: 27 MHz
    parameter BAUD           = 115200,
    parameter LED_ACTIVE_LOW = 1             // LEDs de la Tang Nano 20K: activos en bajo
)(
    input  wire clk,
    input  wire uart_rxd,             // pin RX de la FPGA (viene del PC / Arduino)
    output wire uart_txd,             // pin TX de la FPGA (va al PC / Arduino)
    output wire led_done,             // encendido = crecimiento terminado
    output wire led_run               // encendido = corriendo libre
);
    localparam FRAC   = 16;
    localparam W      = 24;
    localparam SHIFT  = 3;
    localparam NB     = N*W + N + 1;
    localparam NBYTES = (NB + 7) / 8;
    localparam HDR    = 4;                    // 0xA5, N, N, A

    // los 4 movimientos (constantes de construccion)
    function integer modinv;
        input integer a_in;
        input integer m;
        integer k;
        begin
            modinv = 0;
            for (k = 1; k < m; k = k + 1)
                if (((a_in * k) % m) == 1) modinv = k;
        end
    endfunction
    localparam AM    = A % N;
    localparam AINV  = modinv(AM, N);
    localparam A2    = (AM   * AM)   % N;
    localparam A2INV = (AINV * AINV) % N;

    // reset de encendido (~16 ciclos)
    reg [3:0] por;
    initial por = 4'd0;
    wire por_busy = ~&por;
    always @(posedge clk) if (por_busy) por <= por + 4'd1;

    reg cmd_rst, run, step, sending, tx_start;
    reg [7:0]  tx_data;
    reg [15:0] idx;

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

    // ---------------- foto binaria ----------------
    wire [NBYTES*8-1:0] vec = {v_flat, active, done};
    reg  [7:0] bin_byte;
    always @* begin
        case (idx)
            16'd0:   bin_byte = 8'hA5;
            16'd1:   bin_byte = N;
            16'd2:   bin_byte = N;
            16'd3:   bin_byte = AM;
            default: bin_byte = vec[8*(idx-HDR) +: 8];
        endcase
    end

    // ---------------- conversion a decimal ----------------
    function [11:0] bcd3;                     // 8 bits -> 3 digitos BCD
        input [7:0] v;
        integer i;
        reg [19:0] s;
        begin
            s = {12'd0, v};
            for (i = 0; i < 8; i = i + 1) begin
                if (s[11:8]  >= 5) s[11:8]  = s[11:8]  + 4'd3;
                if (s[15:12] >= 5) s[15:12] = s[15:12] + 4'd3;
                if (s[19:16] >= 5) s[19:16] = s[19:16] + 4'd3;
                s = s << 1;
            end
            bcd3 = s[19:8];
        end
    endfunction

    // digito pos (0 = centenas) sin ceros a la izquierda -> {saltar, caracter}
    function [8:0] dig;
        input [11:0] b;
        input [1:0]  pos;
        begin
            case (pos)
                2'd0:    dig = {(b[11:8] == 0),                4'h3, b[11:8]};
                2'd1:    dig = {(b[11:8] == 0 && b[7:4] == 0), 4'h3, b[7:4]};
                default: dig = {1'b0,                          4'h3, b[3:0]};
            endcase
        end
    endfunction

    // ---------------- volcado de texto ----------------
    localparam [3:0] D_HDR0 = 0, D_HDR = 1, D_MOV = 2, D_DIV0 = 3, D_DIV = 4,
                     D_DD = 5, D_INV = 6, D_FIND = 7, D_PREP = 8, D_FRAC = 9, D_CELL = 10;
    localparam [1:0] M_BIN = 0, M_DUMP = 1;

    reg [1:0]  mode;
    reg [3:0]  ds;
    reg [4:0]  lp;                    // posicion dentro de la linea
    reg [7:0]  cx;                    // celda
    reg [11:0] bn, ba, bo, bx, bi, bm0, bm1, bm2, bm3;
    reg        bd, dneg, inf;
    reg [W-1:0] dv;
    reg [15:0] ff, fdig;
    reg [4:0]  it;                    // iteraciones del divisor / double dabble
    reg [23:0] dnd, quo;              // dividendo, cociente
    reg [24:0] rem;
    reg [23:0] dvs;                   // divisor = V1
    reg [31:0] bcd8;                  // 8 digitos de round(100 / V1)

    initial begin
        cmd_rst = 0; run = 0; step = 0; sending = 0; tx_start = 0; tx_data = 0; idx = 0;
        mode = M_BIN; ds = 0; lp = 0; cx = 0;
    end

    // "n=21 a=2 done=1 active=6\r\n"    -> {ultimo, saltar, caracter}
    reg [9:0] hdr_ch;
    always @* begin
        case (lp)
            5'd0:  hdr_ch = {2'b00, "n"};
            5'd1:  hdr_ch = {2'b00, "="};
            5'd2:  hdr_ch = {1'b0, dig(bn, 2'd0)};
            5'd3:  hdr_ch = {1'b0, dig(bn, 2'd1)};
            5'd4:  hdr_ch = {1'b0, dig(bn, 2'd2)};
            5'd5:  hdr_ch = {2'b00, " "};
            5'd6:  hdr_ch = {2'b00, "a"};
            5'd7:  hdr_ch = {2'b00, "="};
            5'd8:  hdr_ch = {1'b0, dig(ba, 2'd0)};
            5'd9:  hdr_ch = {1'b0, dig(ba, 2'd1)};
            5'd10: hdr_ch = {1'b0, dig(ba, 2'd2)};
            5'd11: hdr_ch = {2'b00, " "};
            5'd12: hdr_ch = {2'b00, "d"};
            5'd13: hdr_ch = {2'b00, "o"};
            5'd14: hdr_ch = {2'b00, "n"};
            5'd15: hdr_ch = {2'b00, "e"};
            5'd16: hdr_ch = {2'b00, "="};
            5'd17: hdr_ch = {2'b00, bd ? "1" : "0"};
            5'd18: hdr_ch = {2'b00, " "};
            5'd19: hdr_ch = {2'b00, "a"};
            5'd20: hdr_ch = {2'b00, "c"};
            5'd21: hdr_ch = {2'b00, "t"};
            5'd22: hdr_ch = {2'b00, "i"};
            5'd23: hdr_ch = {2'b00, "v"};
            5'd24: hdr_ch = {2'b00, "e"};
            5'd25: hdr_ch = {2'b00, "="};
            5'd26: hdr_ch = {1'b0, dig(bo, 2'd0)};
            5'd27: hdr_ch = {1'b0, dig(bo, 2'd1)};
            5'd28: hdr_ch = {1'b0, dig(bo, 2'd2)};
            5'd29: hdr_ch = {2'b00, 8'h0D};
            default: hdr_ch = {2'b10, 8'h0A};
        endcase
    end

    // "moves: x2 x11 x4 x16\r\n"
    reg [9:0] mov_ch;
    always @* begin
        case (lp)
            5'd0:  mov_ch = {2'b00, "m"};
            5'd1:  mov_ch = {2'b00, "o"};
            5'd2:  mov_ch = {2'b00, "v"};
            5'd3:  mov_ch = {2'b00, "e"};
            5'd4:  mov_ch = {2'b00, "s"};
            5'd5:  mov_ch = {2'b00, ":"};
            5'd6:  mov_ch = {2'b00, " "};
            5'd7:  mov_ch = {2'b00, "x"};
            5'd8:  mov_ch = {1'b0, dig(bm0, 2'd0)};
            5'd9:  mov_ch = {1'b0, dig(bm0, 2'd1)};
            5'd10: mov_ch = {1'b0, dig(bm0, 2'd2)};
            5'd11: mov_ch = {2'b00, " "};
            5'd12: mov_ch = {2'b00, "x"};
            5'd13: mov_ch = {1'b0, dig(bm1, 2'd0)};
            5'd14: mov_ch = {1'b0, dig(bm1, 2'd1)};
            5'd15: mov_ch = {1'b0, dig(bm1, 2'd2)};
            5'd16: mov_ch = {2'b00, " "};
            5'd17: mov_ch = {2'b00, "x"};
            5'd18: mov_ch = {1'b0, dig(bm2, 2'd0)};
            5'd19: mov_ch = {1'b0, dig(bm2, 2'd1)};
            5'd20: mov_ch = {1'b0, dig(bm2, 2'd2)};
            5'd21: mov_ch = {2'b00, " "};
            5'd22: mov_ch = {2'b00, "x"};
            5'd23: mov_ch = {1'b0, dig(bm3, 2'd0)};
            5'd24: mov_ch = {1'b0, dig(bm3, 2'd1)};
            5'd25: mov_ch = {1'b0, dig(bm3, 2'd2)};
            5'd26: mov_ch = {2'b00, 8'h0D};
            default: mov_ch = {2'b10, 8'h0A};
        endcase
    end

    // "order from V1: 1/v1=6.00\r\n"  (o "1/v1=inf")
    localparam [8*20-1:0] L3 = "order from V1: 1/v1=";
    reg [9:0] inv_ch;
    always @* begin
        if (lp < 5'd20)
            inv_ch = {2'b00, L3[8*(19-lp) +: 8]};
        else if (inf) case (lp)
            5'd20: inv_ch = {2'b00, "i"};
            5'd21: inv_ch = {2'b00, "n"};
            5'd22: inv_ch = {2'b00, "f"};
            5'd23: inv_ch = {2'b00, 8'h0D};
            default: inv_ch = {2'b10, 8'h0A};
        endcase else case (lp)
            // digitos 7..2 = parte entera (sin ceros a la izquierda), luego . d1 d0
            5'd20: inv_ch = {1'b0, (bcd8[31:28] == 0), 4'h3, bcd8[31:28]};
            5'd21: inv_ch = {1'b0, (bcd8[31:24] == 0), 4'h3, bcd8[27:24]};
            5'd22: inv_ch = {1'b0, (bcd8[31:20] == 0), 4'h3, bcd8[23:20]};
            5'd23: inv_ch = {1'b0, (bcd8[31:16] == 0), 4'h3, bcd8[19:16]};
            5'd24: inv_ch = {1'b0, (bcd8[31:12] == 0), 4'h3, bcd8[15:12]};
            5'd25: inv_ch = {2'b00, 4'h3, bcd8[11:8]};
            5'd26: inv_ch = {2'b00, "."};
            5'd27: inv_ch = {2'b00, 4'h3, bcd8[7:4]};
            5'd28: inv_ch = {2'b00, 4'h3, bcd8[3:0]};
            5'd29: inv_ch = {2'b00, 8'h0D};
            default: inv_ch = {2'b10, 8'h0A};
        endcase
    end

    // "x=1 v=0.1667\r\n"
    reg [9:0] cell_ch;
    always @* begin
        case (lp)
            5'd0:  cell_ch = {2'b00, "x"};
            5'd1:  cell_ch = {2'b00, "="};
            5'd2:  cell_ch = {1'b0, dig(bx, 2'd0)};
            5'd3:  cell_ch = {1'b0, dig(bx, 2'd1)};
            5'd4:  cell_ch = {1'b0, dig(bx, 2'd2)};
            5'd5:  cell_ch = {2'b00, " "};
            5'd6:  cell_ch = {2'b00, "v"};
            5'd7:  cell_ch = {2'b00, "="};
            5'd8:  cell_ch = {1'b0, ~dneg, "-"};
            5'd9:  cell_ch = {1'b0, dig(bi, 2'd0)};
            5'd10: cell_ch = {1'b0, dig(bi, 2'd1)};
            5'd11: cell_ch = {1'b0, dig(bi, 2'd2)};
            5'd12: cell_ch = {2'b00, "."};
            5'd13: cell_ch = {2'b00, 4'h3, fdig[15:12]};
            5'd14: cell_ch = {2'b00, 4'h3, fdig[11:8]};
            5'd15: cell_ch = {2'b00, 4'h3, fdig[7:4]};
            5'd16: cell_ch = {2'b00, 4'h3, fdig[3:0]};
            5'd17: cell_ch = {2'b00, 8'h0D};
            default: cell_ch = {2'b10, 8'h0A};
        endcase
    end

    reg [9:0] line_ch;
    always @* begin
        case (ds)
            D_HDR:   line_ch = hdr_ch;
            D_MOV:   line_ch = mov_ch;
            D_INV:   line_ch = inv_ch;
            default: line_ch = cell_ch;
        endcase
    end

    wire signed [W-1:0] vcx  = v_flat[cx*W +: W];
    wire signed [W-1:0] v1   = v_flat[1*W +: W];
    wire [19:0]         ff10 = {ff, 3'b000} + {ff, 1'b0};      // ff * 10
    wire [24:0]         rsh  = {rem[23:0], dnd[23]};           // paso del divisor

    // double dabble: suma 3 a cada digito >= 5
    function [31:0] add3;
        input [31:0] b;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                add3[4*i +: 4] = (b[4*i +: 4] >= 5) ? b[4*i +: 4] + 4'd3 : b[4*i +: 4];
        end
    endfunction
    wire [31:0] bcd_adj = add3(bcd8);

    always @(posedge clk) begin
        cmd_rst  <= 1'b0;
        step     <= 1'b0;
        tx_start <= 1'b0;

        // ---- comandos ----
        if (rx_valid && !sending) begin
            case (rx_data)
                "R", "r": begin cmd_rst <= 1'b1; run <= 1'b0; end
                "T", "t": step <= 1'b1;
                "G", "g": run  <= 1'b1;
                "P", "p": run  <= 1'b0;
                "S", "s": begin run <= 1'b0; sending <= 1'b1; mode <= M_BIN;  idx <= 16'd0; end
                "D", "d": begin run <= 1'b0; sending <= 1'b1; mode <= M_DUMP; ds  <= D_HDR0; end
                default: ;
            endcase
        end

        // ---- transmision ----
        if (sending && !tx_busy && !tx_start) begin
            if (mode == M_BIN) begin
                tx_data <= bin_byte; tx_start <= 1'b1;
                if (idx == HDR + NBYTES - 1) sending <= 1'b0;
                else                         idx <= idx + 16'd1;
            end else case (ds)
                D_HDR0: begin
                    bn  <= bcd3(N);     ba  <= bcd3(AM);
                    bm0 <= bcd3(AM);    bm1 <= bcd3(AINV);
                    bm2 <= bcd3(A2);    bm3 <= bcd3(A2INV);
                    bo  <= bcd3(order[7:0]);  bd <= done;
                    lp  <= 0;  ds <= D_HDR;
                end
                D_HDR, D_MOV, D_INV, D_CELL: begin
                    if (!line_ch[8]) begin tx_data <= line_ch[7:0]; tx_start <= 1'b1; end
                    if (line_ch[9]) begin
                        lp <= 0;
                        case (ds)
                            D_HDR:   ds <= D_MOV;
                            D_MOV:   ds <= D_DIV0;
                            D_INV:   begin cx <= 0;          ds <= D_FIND; end
                            default: begin cx <= cx + 8'd1;  ds <= D_FIND; end
                        endcase
                    end else lp <= lp + 5'd1;
                end
                // ---- 1/V1 :  round(100 * 2^FRAC / V1) con un divisor secuencial ----
                D_DIV0: begin
                    inf  <= (v1 <= 0);
                    dvs  <= v1;
                    dnd  <= (100 << FRAC) + (v1 >>> 1);
                    rem  <= 0;  quo <= 0;  it <= 0;
                    ds   <= (v1 <= 0) ? D_INV : D_DIV;
                end
                D_DIV: begin
                    if (rsh >= {1'b0, dvs}) begin
                        rem <= rsh - {1'b0, dvs};  quo <= {quo[22:0], 1'b1};
                    end else begin
                        rem <= rsh;                quo <= {quo[22:0], 1'b0};
                    end
                    dnd <= {dnd[22:0], 1'b0};
                    it  <= it + 5'd1;
                    if (it == 5'd23) begin bcd8 <= 0; it <= 0; ds <= D_DD; end
                end
                D_DD: begin                         // quo -> 8 digitos BCD
                    if (it == 5'd24) ds <= D_INV;
                    else begin
                        bcd8 <= {bcd_adj[30:0], quo[23]};
                        quo  <= {quo[22:0], 1'b0};
                        it   <= it + 5'd1;
                    end
                end
                // ---- una linea por celda activa ----
                D_FIND: begin
                    if (cx >= N)          sending <= 1'b0;     // fin del volcado
                    else if (active[cx]) begin
                        dneg <= vcx[W-1];
                        dv   <= vcx[W-1] ? -vcx : vcx;
                        ds   <= D_PREP;
                    end else cx <= cx + 8'd1;
                end
                D_PREP: begin
                    bx <= bcd3(cx);
                    bi <= bcd3(dv[W-1:FRAC]);
                    ff <= dv[FRAC-1:0];
                    it <= 0;
                    ds <= D_FRAC;
                end
                default: begin                      // D_FRAC: 4 decimales
                    fdig <= {fdig[11:0], ff10[19:16]};
                    ff   <= ff10[15:0];
                    it   <= it + 5'd1;
                    if (it == 5'd3) begin lp <= 0; ds <= D_CELL; end
                end
            endcase
        end
    end

    assign led_done = done ^ LED_ACTIVE_LOW[0];
    assign led_run  = run  ^ LED_ACTIVE_LOW[0];
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
//  tb_uart : simula un PC / Arduino que manda comandos por UART
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
    localparam HDR    = 4;

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

    // ---- FPGA -> PC ----
    reg [7:0] rxbuf [0:1023];
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

    task wait_quiet;
        integer last;
        begin
            last = -1;
            while (last != rxn) begin
                last = rxn;
                repeat (20*CPB) @(posedge clk);
            end
        end
    endtask

    // busca el texto s (len caracteres) en lo recibido
    integer errors;
    task expect_text;
        input [8*32-1:0] s;
        input integer    len;
        integer p, q, ok, found;
        begin
            found = 0;
            for (p = 0; p + len <= rxn; p = p + 1) begin
                ok = 1;
                for (q = 0; q < len; q = q + 1)
                    if (rxbuf[p+q] !== s[8*(len-1-q) +: 8]) ok = 0;
                if (ok) found = 1;
            end
            if (!found) begin
                $display("  !! no aparece el texto \"%0s\"", s);
                errors = errors + 1;
            end
        end
    endtask

    // ---- foto binaria ----
    reg [NBYTES*8-1:0] vec;
    reg signed [W-1:0] vx;
    integer x, i, t, nact, total, tol;
    reg     done_f;

    task snapshot;
        begin
            rxn = 0;
            send_byte("S");
            wait (rxn == HDR + NBYTES);
            repeat (2*CPB) @(posedge clk);
            if (rxbuf[0] !== 8'hA5 || rxbuf[1] !== N || rxbuf[2] !== N || rxbuf[3] !== A % N) begin
                $display("  !! cabecera incorrecta: %h %h %h %h", rxbuf[0], rxbuf[1], rxbuf[2], rxbuf[3]);
                errors = errors + 1;
            end
            for (i = 0; i < NBYTES; i = i + 1) vec[8*i +: 8] = rxbuf[HDR+i];
            done_f = vec[0];
            nact = 0; total = 0;
            for (x = 0; x < N; x = x + 1) begin
                vx = vec[N+1 + x*W +: W];
                total = total + vx;
                if (vec[1+x]) nact = nact + 1;
            end
        end
    endtask

    task dump;
        begin
            rxn = 0;
            send_byte("D");
            wait_quiet;
            for (i = 0; i < rxn; i = i + 1) if (rxbuf[i] != 8'h0D) $write("%c", rxbuf[i]);
        end
    endtask

    initial begin
        errors = 0;
        repeat (50) @(posedge clk);

        $display("--- ronda a ronda: R, luego T + S ---");
        send_byte("R");
        dump;
        expect_text("n=21 a=2 done=0 active=1", 24);
        expect_text("moves: x2 x11 x4 x16", 20);
        expect_text("1/v1=1.00", 9);
        for (t = 1; t <= 4; t = t + 1) begin
            send_byte("T");
            snapshot;
            vx = vec[N+1 + 1*W +: W];
            $display("ronda %0d: activas=%0d done=%0d V1=%8.5f carga=%0d",
                     t, nact, done_f, $itor(vx) / $itor(1 << FRAC), total);
            if (total !== (1 << FRAC)) errors = errors + 1;
            if (t == 2 && (nact !== 6 || done_f !== 1'b1 || vx !== (1 << (FRAC-1)))) begin
                $display("  !! esperaba 6 activas, done=1 y V1=0.5 tras 2 rondas");
                errors = errors + 1;
            end
        end
        dump;
        expect_text("1/v1=4.27", 9);              // V1 = 0.234375 -> 4.2666 -> 4.27

        $display("--- corrida libre: G, espera, S y D ---");
        send_byte("G");
        repeat (200) @(posedge clk);
        snapshot;
        tol = 8 * 6;
        $display("equilibrio: activas=%0d done=%0d carga=%0d (cada una ~%0d)",
                 nact, done_f, total, (1 << FRAC) / 6);
        if (nact !== 6 || total !== (1 << FRAC)) errors = errors + 1;
        for (x = 0; x < N; x = x + 1) begin
            vx = vec[N+1 + x*W +: W];
            if (vec[1+x] && ((vx - (1 << FRAC)/6 > tol) || ((1 << FRAC)/6 - vx > tol))) begin
                $display("  celda %0d fuera de tolerancia: %0d", x, vx);
                errors = errors + 1;
            end
        end
        dump;
        expect_text("done=1 active=6", 15);
        expect_text("1/v1=6.0", 8);

        if (errors == 0) $display("RESULTADO UART: PASS");
        else             $display("RESULTADO UART: FAIL (%0d errores)", errors);
        $finish;
    end
endmodule
`endif
