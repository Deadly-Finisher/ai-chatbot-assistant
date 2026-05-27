const fs = require('fs-extra');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const DATA_DIR = path.join(__dirname, '..', 'data');

fs.ensureDirSync(DATA_DIR);

const USERS_FILE =
    path.join(DATA_DIR, 'users.json');

class UserRepository {

    constructor() {

        if (!fs.existsSync(USERS_FILE)) {

            fs.writeJSONSync(
                USERS_FILE,
                { users: [] },
                { spaces: 2 }
            );
        }

        this._load();
    }

    _load() {

        const data =
            fs.readJSONSync(USERS_FILE);

        this.users = data.users || [];
    }

    _save() {

        fs.writeJSONSync(
            USERS_FILE,
            { users: this.users },
            { spaces: 2 }
        );
    }

    createUser(username, password) {

        const user = {

            id: uuidv4(),

            username,

            password,

            createdAt:
                new Date().toISOString()
        };

        this.users.push(user);

        this._save();

        return {
            id: user.id,
            username: user.username
        };
    }

    findByUsername(username) {

        return this.users.find(
            u => u.username === username
        );
    }

    findById(id) {

        return this.users.find(
            u => u.id === id
        );
    }
}

module.exports = UserRepository;