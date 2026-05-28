const fs = require('fs-extra');
const path = require('path');

class TaskRepository {

    constructor() {

        this.file =
            path.join(
                __dirname,
                '../data/tasks.json'
            );

        fs.ensureFileSync(
            this.file
        );

        if (
            fs.readFileSync(
                this.file,
                'utf8'
            ).trim() === ''
        ) {

            fs.writeJsonSync(
                this.file,
                []
            );
        }
    }

    getAllTasks() {

        return fs.readJsonSync(
            this.file
        );
    }

    saveTasks(tasks) {

        fs.writeJsonSync(
            this.file,
            tasks,
            { spaces: 2 }
        );
    }

    addTask(task) {

        const tasks =
            this.getAllTasks();

        tasks.push(task);

        this.saveTasks(tasks);

        return task;
    }

    getUserTasks(userId) {

        return this
            .getAllTasks()
            .filter(
                t => t.userId === userId
            );
    }

    updateTask(taskId, updates) {

        const tasks =
            this.getAllTasks();

        const index =
            tasks.findIndex(
                t => t.id === taskId
            );

        if (index === -1)
            return null;

        tasks[index] = {

            ...tasks[index],

            ...updates
        };

        this.saveTasks(tasks);

        return tasks[index];
    }
}

module.exports =
    TaskRepository;