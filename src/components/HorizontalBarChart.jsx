import React from "react";

function HorizontalBarChart({ predictionArray, labels}) {
  return (
    <div>
      {predictionArray.map((confidence, index) => (
        <div key={index}>

<div class="flex justify-between mb-1">
  <span class="text-base font-medium">{labels[index].name}</span>
  <span class="text-sm font-medium ">{ `${confidence * 100}%` }</span>
</div>
<div class="w-full bg-slate-200 rounded-full h-2.5">
  <div class="bg-blue-600 h-2.5 rounded-full" style={{ width: `${confidence * 100}%` }}></div>
</div>
        </div>
      ))}
    </div>
  );
}

export default HorizontalBarChart;
