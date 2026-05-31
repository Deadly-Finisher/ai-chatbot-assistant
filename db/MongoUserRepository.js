const User =
    require('../models/user');

class MongoUserRepository {

    async createUser(
        username,
        password
    ) {

        const user =
            await User.create({

                username,
                password

            });

        return {

            id: user.id.toString(),

            username:
                user.username
        };
    }

    async findByUsername(
        username
    ) {

        return await User.findOne({
            username
        });
    }

    async findById(id) {

        return await User.findById(id);
    }
}

module.exports =
    MongoUserRepository;