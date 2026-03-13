// script.js

// Function to add a new todo item
function addTodo(item) {
    if (!item || item.trim() === '') {
        alert('Please enter a valid todo item.');
        return;
    }
    const todos = getTodosFromLocalStorage();
    const newTodo = { id: Date.now(), item: item, completed: false };
    todos.push(newTodo);
    saveTodosToLocalStorage(todos);
    renderTodos(todos);
}

// Function to mark a todo item as completed
function markTodoCompleted(id) {
    const todos = getTodosFromLocalStorage();
    const todo = todos.find(todo => todo.id === id);
    if (todo) {
        todo.completed = !todo.completed;
        saveTodosToLocalStorage(todos);
        renderTodos(todos);
    }
}

// Function to delete a todo item
function deleteTodo(id) {
    let todos = getTodosFromLocalStorage();
    todos = todos.filter(todo => todo.id !== id);
    saveTodosToLocalStorage(todos);
    renderTodos(todos);
}

// Function to filter todos based on the filter criteria
function filterTodos(filter) {
    const todos = getTodosFromLocalStorage();
    let filteredTodos;
    if (filter === 'completed') {
        filteredTodos = todos.filter(todo => todo.completed);
    } else if (filter === 'active') {
        filteredTodos = todos.filter(todo => !todo.completed);
    } else {
        filteredTodos = todos;
    }
    renderTodos(filteredTodos);
}

// Function to persist todos to local storage
function persistTodos() {
    const todos = getTodosFromLocalStorage();
    saveTodosToLocalStorage(todos);
}

// Function to load todos from local storage on initialization
function loadTodos() {
    const todos = getTodosFromLocalStorage();
    renderTodos(todos);
}

// Helper function to get todos from local storage
function getTodosFromLocalStorage() {
    const todos = localStorage.getItem('todos');
    return todos ? JSON.parse(todos) : [];
}

// Helper function to save todos to local storage
function saveTodosToLocalStorage(todos) {
    localStorage.setItem('todos', JSON.stringify(todos));
}

// Function to render todos in the DOM
function renderTodos(todos) {
    const todoList = document.getElementById('todo-list');
    todoList.innerHTML = '';
    if (todos.length === 0) {
        todoList.innerHTML = '<li>No todos available.</li>';
        return;
    }
    todos.forEach(todo => {
        const todoItem = document.createElement('li');
        todoItem.textContent = todo.item;
        if (todo.completed) {
            todoItem.style.textDecoration = 'line-through';
        }
        todoItem.addEventListener('click', () => markTodoCompleted(todo.id));
        const deleteButton = document.createElement('button');
        deleteButton.textContent = 'Delete';
        deleteButton.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteTodo(todo.id);
        });
        todoItem.appendChild(deleteButton);
        todoList.appendChild(todoItem);
    });
}

// Load todos when the window is loaded
window.onload = loadTodos;