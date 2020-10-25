const url = "ws://localhost:8765"
var websocket = undefined;
var board_size = [11,11];
var stones = ["●", "○"];

function click_cell(i, j){
    send(JSON.stringify([i,j]))
}

function movein_cell(element, stone){
    element.innerHTML = stone;
    element.classList.add("grid_cell_hovered")
}

function moveout_cell(element){
    element.innerHTML = "";
    element.classList.remove("grid_cell_hovered")
}

function show_board(state){
    var container = document.getElementById("board_container");

    var board = document.createElement("div");
    let h = state.shape[0], w = state.shape[1];
    let stone = stones[state.isNextBlack ? 0:1];
    let board_config = state.board;
    let valid_config = state.validMoves;
    let ended = state.isEnd;

    for(var i=0; i<h; i++){
        var row = document.createElement("div");
        row.className = "grid_row";
        board.appendChild(row);
        for(var j=0; j<w; j++){
            var cell = document.createElement("div");
            cell.className = "grid_cell";
            cell.id = `cell_${i}_${j}`;

            let c = board_config[i][j];
            let v = valid_config[i][j];
            if(c != 0){
                cell.innerHTML = stones[c==1?0:1];
            } else if(v && !ended){
                cell.addEventListener("click", (function(i,j){
                    return function(){
                        click_cell(i,j);
                    };
                })(i,j));
                cell.addEventListener("mouseover", (function(stone){
                    return function(){
                        movein_cell(this, stone);
                    };
                }(stone)));
                cell.addEventListener("mouseout", function(){
                    moveout_cell(this);
                });
            }
            row.appendChild(cell);
        }
    }
    container.innerHTML = "";
    container.appendChild(board);
}

function show_info(state){
    var container = document.getElementById("info");
    if(state.isEnd){
        let result = undefined;
        if(state.endResult == 0){
            result = "tie.";
        } else {
            result = (state.endResult==1? "black":"white") + " won.";
        }
        container.innerHTML = `Game ends. ${result}`;
    }else{
        let c = state.isNextBlack?"black":"white";
        container.innerHTML = `${c}'s move`;
    }
}

function send(data){
    if(websocket){
        websocket.send(data);
    }
}

function reset(){
    if(websocket) websocket.close();
    websocket = new WebSocket(url);
    websocket.addEventListener("message", function(event){
        state = JSON.parse(event.data);
        show_board(state);
        show_info(state);
    });
    websocket.addEventListener("open", function(){
        websocket.send(JSON.stringify(board_size));
    });
}

window.addEventListener("load", function(){
    document.getElementById("reset")
        .addEventListener("click", reset);
});