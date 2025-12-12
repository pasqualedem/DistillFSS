function openDataset(evt, datasetName) {
    var i, x, tablinks;
    x = document.getElementsByClassName("results-table");
    for (i = 0; i < x.length; i++) {
        x[i].style.display = "none";
    }
    tablinks = document.querySelectorAll(".tabs ul li");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace("is-active", "");
    }
    document.getElementById(datasetName).style.display = "table";
    evt.currentTarget.className += " is-active";
}