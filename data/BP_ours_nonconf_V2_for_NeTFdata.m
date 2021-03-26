% load Z_256_BP_nonconf_fixed.mat
% load zaragoza_nonconfocal_Z_256_preprocessed.mat
load conv_S_preprocessed_2.mat
hiddenVolumeSize = hiddenVolumeSize / 4;

NS = 128;
DS = round(256 / NS);
% hiddenVolumeSize = hiddenVolumeSize * 2;
% cameraGridPositions = cameraGridPositions(1:DS:256,1:DS:256,:);
% data = data(1:DS:256,1:DS:256,:);
% laserGridPositions = cameraGridPositions;
% laserGridPositions = laserGridPositions(1:DS:256,1:DS:256,:);
% laserGridPositions = repmat(laserGridPositions, [128, 128, 1]);

%% setting
M = size(data,2);
N = size(data,3);
L = size(data,1);
xmin = hiddenVolumePosition(1) - hiddenVolumeSize / 2;
xmax = hiddenVolumePosition(1) + hiddenVolumeSize / 2;
zmin = hiddenVolumePosition(3) - hiddenVolumeSize / 2;
zmax = hiddenVolumePosition(3) + hiddenVolumeSize / 2;
unit = (xmax - xmin) / (M - 1);
ymax = hiddenVolumePosition(2) + hiddenVolumeSize / 2;
ymin = ymax - unit * (M - 1);

c = 1;
bin = c * deltaT;
% NS = 64; % 采样点数为 NS^2

%% 基球
%{
r0 = 1;
theta = linspace(0, pi, NS);
phi = linspace(-pi, 0, NS);

[Theta,R0,Phi] = meshgrid(theta,r0,phi);
sph_coords = cat(1,R0,Theta,Phi);
sph_coords = reshape(sph_coords,3,[]);
car_coords = zeros(3, NS^2);
car_coords(1,:) = sph_coords(1,:) .* cos(sph_coords(2,:)) .* cos(sph_coords(3,:));
car_coords(2,:) = sph_coords(1,:) .* cos(sph_coords(2,:)) .* sin(sph_coords(3,:));
car_coords(3,:) = sph_coords(1,:) .* sin(sph_coords(2,:));
car_coords = car_coords([1,3,2],:);
car_coords(2,:) = - car_coords(2,:);
% car_coords = car_coords';
%}
x = linspace(xmin,xmax,NS);
z = linspace(zmin,zmax,NS);
[Z,X] = meshgrid(z,x);
car_coords = zeros(3,NS^2);
car_coords(1,:) = reshape(X, 1,[]);
car_coords(3,:) = reshape(Z, 1,[]);
% car_coords(2,:) 需要在每个循环中单独计算


% scatter3(car_coords(1,:),car_coords(2,:),car_coords(3,:))
% xlabel('X')
% ylabel('Y')
% zlabel('Z')

%%
value = zeros(NS, N, NS);

for i = 1:4:M
    for j = 1:4:N
%         i = 32;
%         j = 32;
%         temp = cell(L, 2);
%         temp_index = zeros(L,NS^2);
%         temp_data = zeros(L,NS^2);
        for k = 1:1:L

%               k = 710;
%             disp(k)
%             disp(data(i,j,k))
            if data(k,i,j) == 0
                continue
            end
%             if data(k,i,j) ~= max(data(:,i,j),[],'all')
%                 continue
%             end  
            
%             disp(k)
            CGP = reshape(cameraGridPositions(:,(i-1) * N + j),3,1);
            LGP = reshape(laserGridPositions(:,(i-1) * N + j),3,1);
%             CGP = reshape(cameraGridPositions(:,(i - 1) * N + j),3,1);
%             LGP = reshape(laserGridPositions(:,(i - 1) * N + j),3,1);
            center = (CGP + LGP) / 2;
            vector = (CGP - LGP);
            vector = vector([1,3],1);
            vector = vector(1) + 1i * vector(2);
            
            OL = bin * k;
            a = bin * k / 2;
            f = norm(CGP - LGP) / 2;
            b = sqrt(a^2 - f^2);
            if ~isreal(b)
                continue
            end
%             pts = car_coords;
%             pts(1,:) = pts(1,:) * a;
%             pts(2,:) = pts(2,:) * b;
%             pts(3,:) = pts(3,:) * b;
%             pts = car_coords * r;
%             pts = pts + reshape(cameraGridPositions(i,j,:),3,1);

%             rotation_angle = angle(vector);
%             rotation_matrix = [cos(rotation_angle),- sin(rotation_angle);sin(rotation_angle), cos(rotation_angle)];
%             pts_rotated_xz = rotation_matrix * pts([1,3],:);
%             pts(1,:) = pts_rotated_xz(1,:);
%             pts(3,:) = pts_rotated_xz(2,:);
%             
%             pts(1,:) = pts(1,:) + center(1);
%             pts(3,:) = pts(3,:) + center(3);

            rotation_angle = angle(vector);
            rotation_angle = - rotation_angle;
            ct = cos(rotation_angle);
            st = sin(rotation_angle);
            pts = car_coords;
            sx = pts(1,:) - center(1);
            sz = pts(3,:) - center(3);
            
            y_n = 1 - ((ct * sx - st * sz) / a).^2 - ((st * sx + ct * sz) / b).^2 ;
            pts = pts(:,y_n > 0);
            y_n = y_n(y_n > 0);
            pts(2,:) =  -b * sqrt(y_n);
            
%             scatter3(pts(1,:),pts(2,:),pts(3,:))
%             hold on
%             scatter3(CGP(1),CGP(2),CGP(3),'g')
%             scatter3(LGP(1),LGP(2),LGP(3),'r')
%             xlim([-1,1])
%             ylim([-1.5,0.5])
%             zlim([-1,1])
%             hold off
%             CGP
%             LGP
%             pause(0.1)


            index = zeros(3,size(pts,2));
            index(1,:) = (pts(1,:) - xmin) / (xmax - xmin) * NS;
            index(2,:) = (pts(2,:) - ymin) / (ymax - ymin) * N;
            index(3,:) = (pts(3,:) - zmin) / (zmax - zmin) * NS;
            % 到此处，index的每个元素是一个椭球面上每个点每个方向上在volume中的小数索引
            index_index = (index(1,:) > 0.5) .* (index(2,:) > 0.5) .* (index(3,:) > 0.5) .* (index(1,:) < M + 0.5) .* (index(2,:) < M + 0.5) .* (index(3,:) < M + 0.5);
            index_index = logical(index_index);
            index = index(:,index_index);
            index = round(index);
            index = sub2ind([NS,N,NS],index(1,:),index(2,:),index(3,:));
            value(index) = value(index) + data(k,i,j);
%             temp{k,1} = index;
%             temp{k,2} = data(i,j,k);
%             temp_index(k,:) = index;
%             temp_data(k,:) = data(i,j,k);
        end
%         disp(j)
        
%         temp_index = temp_index(:);
%         temp_data = temp_data(:);
        
    end
    disp(i)
end
%%
% load value.mat
volume = value;%(1:4:end,:,1:4:end);
% h = - fspecial('laplacian',0);
I = volume;
[map, depth] = max(I,[],2);
map = squeeze(map);
% map = imfilter(map, h);
imshow( 1 * map / max(map,[],'all'),'InitialMagnification','fit')
% save value_S_128.mat value
% value = ...
% value(1:2:end,1:2:end,1:2:end) + value(1:2:end,1:2:end,2:2:end) + ...
% value(1:2:end,2:2:end,1:2:end) + value(1:2:end,2:2:end,2:2:end) + ...
% value(2:2:end,1:2:end,1:2:end) + value(2:2:end,1:2:end,2:2:end) + ...
% value(2:2:end,2:2:end,1:2:end) + value(2:2:end,2:2:end,2:2:end);




