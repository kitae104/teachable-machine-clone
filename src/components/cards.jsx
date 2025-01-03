import React from 'react';

const Card = ({ id, title, description }) => {
    return (
        <div id={id} className="bg-white rounded-lg shadow-md p-4">
            <h2 className="text-xl font-medium mb-2">{title}</h2>
            <p className="text-gray-700">{description}</p>
        </div>
    );
};

export default Card;