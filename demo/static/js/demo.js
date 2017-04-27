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
            xhr.open("GET", "/classify/" + encodeURIComponent(text), true);
            xhr.send();
        });
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

        classify(userText.value).then((result) => {
            ellipses.innerHTML = "";
            clearInterval(progress);
            var results = $("classifier-results");
            results.style.display = "block";
            var table = results.querySelector("tbody");
            var h2 = results.querySelector("h2");
            var maxClass;
            table.innerHTML = "";
            var sorted = [];
            for (let key in result) {
                if (!maxClass || result[key] > result[maxClass]) {
                    maxClass = key;
                }
                sorted.push([key, result[key]])
            }
            sorted = sorted.sort((a, b) => b[1] - a[1]);

            h2.textContent = maxClass;
            for (var i = 0; i < sorted.length; i++) {
                let key = sorted[i][0]
                let value = sorted[i][1]
                let tr = document.createElement("tr");
                let th = document.createElement("th");
                let td = document.createElement("td");
                let bar = document.createElement("span");
                bar.classList.add("bar");
                th.textContent = key;
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

        });
    });
});
