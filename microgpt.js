/*-----
这是对 Andrej Karpathy “原子级” Python 实现的直接移植。旨在仅使用 Node.js 标准库 + ES5语法，演示 Transformer 的核心算法——自动求导、注意力机制和优化器。
------*/

var fs = require('fs');
var path = require('path');
var child_process = require('child_process');

var _seed = 42;
function rand() {
    _seed = (_seed * 16807) % 2147483647;
    return (_seed - 1) / 2147483646;
}

function randn() {
    var u = rand();
    var v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

//数据加载（对应 Python os + urllib）
var inputPath = path.join(__dirname, 'input.txt');
//找不到数据文件，直接下载
if (!fs.existsSync(inputPath)) {
    console.log('input.txt not found, downloading from GitHub...');
    var url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt';
    // 使用 curl 同步下载
    child_process.execSync('curl -L ' + url + ' -o ' + inputPath, { stdio: 'inherit' });
    console.log('Download finished.');
}


var text = fs.readFileSync(inputPath, 'utf8');
var lines = text.split('\n');

var docs = [];
for (var i = 0; i < lines.length; i++) {
    var s = lines[i].trim();
    if (s.length > 0) {
        docs.push(s);
    }
}

//shuffle
for (var i = docs.length - 1; i > 0; i--) {
    var j = Math.floor(rand() * (i + 1));
    var tmp = docs[i];
    docs[i] = docs[j];
    docs[j] = tmp;
}

console.log('num docs:', docs.length);


//Tokenizer
var chars = {};
for (var i = 0; i < docs.length; i++) {
    var d = docs[i];
    for (var k = 0; k < d.length; k++) {
        chars[d[k]] = true;
    }
}

var uchars = [];
for (var c in chars){
    uchars.push(c);
}
uchars.sort();

var BOS = uchars.length;
var vocabSize = uchars.length + 1;

function charToId(ch) {
    for (var i = 0; i < uchars.length; i++) {
        if (uchars[i] === ch){
            return i;
        }
    }
    return -1;
}

function idToChar(id) {
    return uchars[id];
}

console.log('vocab size:', vocabSize);


//自动求导 Value（完全等价）
function Value(data, children, grads) {
    this.data = data;
    this.grad = 0;
    this.children = children || [];
    this.grads = grads || [];
}

function add(a, b) {
    return new Value(a.data + b.data, [a, b], [1, 1]);
}

function mul(a, b) {
    return new Value(a.data * b.data, [a, b], [b.data, a.data]);
}

function neg(a) {
    return mul(a, new Value(-1));
}

function sub(a, b) {
    return add(a, neg(b));
}

function pow(a, p) {
    return new Value(Math.pow(a.data, p), [a], [p * Math.pow(a.data, p - 1)]);
}

function logv(a) {
    return new Value(Math.log(a.data), [a], [1 / a.data]);
}

function expv(a) {
    var e = Math.exp(a.data);
    return new Value(e, [a], [e]);
}

function relu(a) {
    return new Value(a.data > 0 ? a.data : 0, [a], [a.data > 0 ? 1 : 0]);
}



//backward（反向传播）
function backward(v) {
    var topo = [];
    var visited = [];

    function visit(x) {
        if (visited.indexOf(x) !== -1){
            return;
        }
        visited.push(x);
        for (var i = 0; i < x.children.length; i++) {
            visit(x.children[i]);
        }
        topo.push(x);
    }

    visit(v);
    v.grad = 1;

    for (var i = topo.length - 1; i >= 0; i--) {
        var node = topo[i];
        for (var j = 0; j < node.children.length; j++) {
            node.children[j].grad += node.grads[j] * node.grad;
        }
    }
}

//模型超参数（与 Python 对齐）
var nLayer = 1;
var nEmb = 16;
var blockSize = 16;
var nHead = 4;
var headDim = nEmb / nHead;


//参数初始化（state_dict）
function matrix(rows, cols) {
    var m = [];
    for (var i = 0; i < rows; i++) {
        var r = [];
        for (var j = 0; j < cols; j++) {
            r.push(new Value(randn() * 0.08));
        }
        m.push(r);
    }
    return m;
}

var state = {
    wte: matrix(vocabSize, nEmb),
    wpe: matrix(blockSize, nEmb),
    lm_head: matrix(vocabSize, nEmb),
    attn_wq: matrix(nEmb, nEmb),
    attn_wk: matrix(nEmb, nEmb),
    attn_wv: matrix(nEmb, nEmb),
    attn_wo: matrix(nEmb, nEmb),
    mlp_fc1: matrix(4 * nEmb, nEmb),
    mlp_fc2: matrix(nEmb, 4 * nEmb)
};


//基础算子（linear / softmax / rmsnorm）
function linear(x, w) {
    var out = [];
    for (var i = 0; i < w.length; i++) {
        var sum = new Value(0);
        for (var j = 0; j < x.length; j++) {
            sum = add(sum, mul(w[i][j], x[j]));
        }
        out.push(sum);
    }
    return out;
}

function softmax(xs) {
    var max = xs[0].data;
    for (var i = 1; i < xs.length; i++) {
        if (xs[i].data > max) max = xs[i].data;
    }

    var exps = [];
    var sum = new Value(0);
    for (var i = 0; i < xs.length; i++) {
        var e = expv(sub(xs[i], new Value(max)));
        exps.push(e);
        sum = add(sum, e);
    }

    var probs = [];
    for (var i = 0; i < exps.length; i++) {
        probs.push(mul(exps[i], pow(sum, -1)));
    }
    return probs;
}

function rmsnorm(x) {
    var sum = new Value(0);
    for (var i = 0; i < x.length; i++) {
        sum = add(sum, mul(x[i], x[i]));
    }
    var mean = mul(sum, new Value(1 / x.length));
    var scale = pow(add(mean, new Value(1e-5)), -0.5);

    var out = [];
    for (var i = 0; i < x.length; i++) {
        out.push(mul(x[i], scale));
    }
    return out;
}


//GPT 前向（完整 Attention + MLP）
function gpt(tokenId, posId, keys, values) {
    var tok = state.wte[tokenId];
    var pos = state.wpe[posId];

    var x = [];
    for (var i = 0; i < nEmb; i++) {
        x.push(add(tok[i], pos[i]));
    }

    x = rmsnorm(x);

    // Attention
    var x_res = x;
    x = rmsnorm(x);

    var q = linear(x, state.attn_wq);
    var k = linear(x, state.attn_wk);
    var v = linear(x, state.attn_wv);

    keys.push(k);
    values.push(v);

    var attn_out = [];
    for (var j = 0; j < nEmb; j++) {
        attn_out.push(new Value(0));
    }

    for (var h = 0; h < nHead; h++) {
        var hs = h * headDim;
        var logits = [];

        for (var t = 0; t < keys.length; t++) {
            var score = new Value(0);
            for (j = 0; j < headDim; j++) {
                score = add(
                    score,
                    mul(q[hs + j], keys[t][hs + j])
                );
            }
            score = mul(score, new Value(1 / Math.sqrt(headDim)));
            logits.push(score);
        }

        var weights = softmax(logits);

        for (var j = 0; j < headDim; j++) {
            var sum = new Value(0);
            for (var t = 0; t < values.length; t++) {
                sum = add(sum, mul(weights[t], values[t][hs + j]));
            }
            attn_out[hs + j] = sum;
        }
    }

    var proj = linear(attn_out, state.attn_wo);
    x = [];
    for (var i = 0; i < nEmb; i++) {
        x.push(add(proj[i], x_res[i]));
    }

    // MLP
    x_res = x;
    x = rmsnorm(x);
    x = linear(x, state.mlp_fc1);
    for (var i = 0; i < x.length; i++){
        x[i] = relu(x[i]);
    }
    x = linear(x, state.mlp_fc2);
    for (var i = 0; i < nEmb; i++) {
        x[i] = add(x[i], x_res[i]);
    }

    return linear(x, state.lm_head);
}


//Adam 优化器（完整）
var lr = 0.01;
var beta1 = 0.85;
var beta2 = 0.99;
var eps = 1e-8;

var params = [];
for (var k in state) {
    var mat = state[k];
    for (var i = 0; i < mat.length; i++) {
        for (var j = 0; j < mat[i].length; j++) {
            params.push(mat[i][j]);
        }
    }
}

var m = [];
var v = [];
for (var i = 0; i < params.length; i++) {
    m.push(0);
    v.push(0);
}


//训练循环
var steps = 1000;

for (var step = 0; step < steps; step++) {
    var doc = docs[step % docs.length];
    var tokens = [BOS];
    for (var i = 0; i < doc.length; i++){
        tokens.push(charToId(doc[i]));
    }
    tokens.push(BOS);

    var keys = [];
    var values = [];
    var loss = new Value(0);

    for (var i = 0; i < tokens.length - 1 && i < blockSize; i++) {
        var logits = gpt(tokens[i], i, keys, values);
        var probs = softmax(logits);
        loss = add(loss, neg(logv(probs[tokens[i + 1]])));
    }

    backward(loss);

    var lr_t = lr * (1 - step / steps);
    for (var i = 0; i < params.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * params[i].grad;
        v[i] = beta2 * v[i] + (1 - beta2) * params[i].grad * params[i].grad;
        var mh = m[i] / (1 - Math.pow(beta1, step + 1));
        var vh = v[i] / (1 - Math.pow(beta2, step + 1));
        params[i].data -= lr_t * mh / (Math.sqrt(vh) + eps);
        params[i].grad = 0;
    }

    if (step % 20 === 0) {
        console.log('step', step, 'loss', loss.data.toFixed(4));
    }
}


//推理生成
console.log('--- inference ---');

for (var s = 0; s < 10; s++) {
    var keys = [];
    var values = [];
    var token = BOS;
    var out = [];

    for (var i = 0; i < blockSize; i++) {
        var logits = gpt(token, i, keys, values);
        var temperature = 0.5;
        var scaledLogits = logits.map(function(l) { 
            return mul(l, new Value(1/temperature)); 
        });
        var probs = softmax(scaledLogits);

        //推理更快等价于temperature = 1
        //var probs = softmax(logits);

        var r = rand();
        var acc = 0;
        var next = BOS;
        for (var j = 0; j < probs.length; j++) {
            acc += probs[j].data;
            if (r < acc) {
                next = j;
                break;
            }
        }

        if (next === BOS) break;
        out.push(idToChar(next));
        token = next;
    }

    console.log(out.join(''));
}
