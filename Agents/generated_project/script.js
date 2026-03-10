let todos = [];

function addTodo(text) {
    if (!text || text.trim() === '') {
        alert('Todo text cannot be empty.');
        return;
    }
    const newTodo = {
        id: Date.now(),
        text: text,
        done: false
    };
    todos.push(newTodo);
    renderTodos();
}

function deleteTodo(id) {
    const index = todos.findIndex(todo => todo.id === id);
    if (index === -1) {
        alert('Todo not found.');
        return;
    }
    todos.splice(index, 1);
    renderTodos();
}

function toggleTodo(id) {
    const todo = todos.find(todo => todo.id === id);
    if (!todo) {
        alert('Todo not found.');
        return;
    }
    todo.done = !todo.done;
    renderTodos();
}

function filterTodos(status) {
    let filteredTodos;
    if (status === 'all') {
        filteredTodos = todos;
    } else if (status === 'completed') {
        filteredTodos = todos.filter(todo => todo.done);
    } else if (status === 'active') {
        filteredTodos = todos.filter(todo => !todo.done);
    } else {
        alert('Invalid filter status.');
        return;
    }
    renderFilteredTodos(filteredTodos);
}

function editTodo(id, newText) {
    const todo = todos.find(todo => todo.id === id);
    if (!todo) {
        alert('Todo not found.');
        return;
    }
    if (!newText || newText.trim() === '') {
        alert('New text cannot be empty.');
        return;
    }
    todo.text = newText;
    renderTodos();
}

function renderTodos() {
    const todoList = document.getElementById('todo-list');
    todoList.innerHTML = '';
    todos.forEach(todo => {
        const todoItem = document.createElement('li');
        todoItem.textContent = todo.text;
        if (todo.done) {
            todoItem.style.textDecoration = 'line-through';
        }
        todoItem.addEventListener('click', () => toggleTodo(todo.id));
        todoList.appendChild(todoItem);
    });
}

function renderFilteredTodos(filteredTodos) {
    const todoList = document.getElementById('todo-list');
    todoList.innerHTML = '';
    filteredTodos.forEach(todo => {
        const todoItem = document.createElement('li');
        todoItem.textContent = todo.text;
        if (todo.done) {
            todoItem.style.textDecoration = 'line-through';
        }
        todoItem.addEventListener('click', () => toggleTodo(todo.id));
        todoList.appendChild(todoItem);
    });
}