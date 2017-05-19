window.addEventListener("load", () => {
    var $ = (x) => document.getElementById(x);


    var userText = $("user-text");
    var submit = $("submit");
    userText.addEventListener("keyup", (e) => {
        if (userText.value) {
            submit.classList.add("active");
        } else {
            submit.classList.remove("active");
            var results = $("classifier-results");
            results.style.display = "none";
        }
        if (e.which == 13) {
            submit.click();
        }
    });

    function request(url, callback) {
        var xhr = new XMLHttpRequest();
        xhr.onload = e => callback(JSON.parse(xhr.response));
        xhr.open("GET", url);
        xhr.send();
    }

    var colors = {
        ANGER:   [255, 32,  88],  // Red
        SADNESS: [59,  163, 252], // Blue
        JOY:     [171, 216, 0]    // Light green
    }

    function rgba(rgb, alpha) {
        return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${alpha})`;
    }

    var classify = (text) => {
        return new Promise((resolve, reject) => {
            var xhr = new XMLHttpRequest();
            xhr.onreadystatechange = function() {
                console.log(this);
                if (this.readyState == 4 && this.status == 200) {
                    console.log(this);
                    resolve(JSON.parse(this.responseText))
                }
            };
            xhr.open("GET", "/classify/" + encodeURIComponent(text.toLowerCase()), true);
            xhr.send();
        });
    }

    var drawBarChart = function(sorted, result, maxClass) {
        var results = $("classifier-results");
        var table = results.querySelector("tbody");

        table.innerHTML = "";

        for (var i = 0; i < sorted.length; i++) {
            let key = sorted[i][0]
            let value = sorted[i][1]
            let tr = document.createElement("tr");
            let th = document.createElement("th");
            let td = document.createElement("td");
            let bar = document.createElement("span");
            bar.classList.add("bar");
            th.textContent = key;
            if (colors[key]) {
                bar.style.backgroundColor = rgba(colors[key], 1);
                th.style.color = rgba(colors[key], 1); 
            }
            bar.textContent = value;
            tr.appendChild(th);
            tr.appendChild(td);
            td.appendChild(bar);
            bar.style.width = Math.round(100 * value / result[maxClass]) + "%";
            if (key == maxClass) {
                bar.classList.add("max-probability-class");
            }
            table.appendChild(tr);
        }
    }

    var playAudio = function(path) {
        var audio = new Audio(path);
        audio.play();
    }

    var drawAttentionChart = function(tokens, attention, maxClass) {
        var parent = $("attention");
        parent.innerHTML = "";
        for (let k = 0; k < tokens.length; k++) {
            let span = document.createElement("span");
            span.textContent = tokens[k];
            span.style.backgroundColor = rgba(colors[maxClass], attention[k]);
            span.className = "token";
            parent.appendChild(span);
            parent.appendChild(document.createTextNode(" "));
        }
    }

    submit.addEventListener("click", () => {
        submit.classList.remove("push");
        submit.classList.add("push");
        var text = userText.value;
        var ellipses = $("loading");
        var progress = window.setInterval(() => {
            ellipses.innerHTML += ".";
            if (ellipses.innerHTML.length > 5) {
                ellipses.innerHTML = ".";
            }
        }, 250);

        classify(userText.value).then((rawdata) => {
            var attention = rawdata.attention;
            var tokens = rawdata.tokens;
            var result = rawdata.classifications;
            var audioPath = rawdata.audio_path;
            var newResult = {};
            for (let key in result) {
                newResult[key] = parseFloat(result[key]);
            }
            console.log(result, newResult);
            result = newResult;
            ellipses.innerHTML = "";
            clearInterval(progress);
            var results = $("classifier-results");
            results.style.display = "block";
            var h2 = results.querySelector("h2");
            var maxClass;
            var sorted = [];
            for (let key in result) {
                if (!maxClass || result[key] > result[maxClass]) {
                    maxClass = key;
                }
                sorted.push([key, result[key]])
            }
            sorted = sorted.sort((a, b) => b[1] - a[1]);
            h2.textContent = maxClass;
            h2.style.color = rgba(colors[maxClass], 1); 
            playAudio(audioPath);
            drawBarChart(sorted, result, maxClass);
            drawAttentionChart(tokens, attention, maxClass);
        });
    });
});
