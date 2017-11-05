
function pLight = fTrafLight(mycolor)
    t = (1/16:1/8:1)'*2*pi;
    x = cos(t);
    y = sin(t);
    pLight = fill(x,y,mycolor);
    axis square;
    axis off;
end