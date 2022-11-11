#=
Simulation based on Physarum Polycephalum to Estimate Optimal Portfolio Allocation
=#

#Startup Code Begin
using Statistics
import StatsBase
import Random
#Startup Code End

function foo()
    n_nutrients = 3
    grid_size = 200
    n_agents = 1000
    step_size = 1
    s_offset = 6
    s_angle = 45
    gradient_decomp_constant = -log(.5)
    deposit_size = 1.1
    nutrient_value = 50.0
    gradient_pct = 0.5
    n_iters=500
    signal_decomp=0.1

        
    return
end

function kernel_filter(trail_grid)
    gsize = size(trail_grid)[1]
    temp_grid = zeros(Float64,gsize+2,gsize+2)
    temp_grid[2:gsize+1,2:gsize+1] = trail_grid
    new = zeros(Float64,gsize,gsize)
    for i in 1:gsize
        temp = temp_grid[:,i:i+2]
        for j in 1:gsize
            kernel = temp[j:j+2,:]
            mu = mean(kernel)
            new[i,j] = mu
        end
    end

    trail_grid = new
    return trail_grid
end



function meshgrid(grid_size)
    x = collect(1:grid_size)' .* ones(grid_size)
    y = ones(grid_size)' .* collect(1:grid_size)
    return x,y
end



function Simulation(
    grid_size::Int64,
    n_agents::Int64,
    s_angle::Int64 = 45,
    s_offset::Int64 = 6,
    signal_decomp::Float64 = 0.1,
    rt_chance::Float64 = 0.05,
    n_iters::Int64 = 5000
    )
    #Set Constants
    deposit_value = 1.10
    drt = rt_chance/(n_agents÷8)

    #Build Grids
    env_grid = zeros(Int64,grid_size,grid_size)
    trail_grid = zeros(Float64,grid_size,grid_size)
    density_grid = zeros(Float64,grid_size,grid_size)

    x0=y0=grid_size÷2 #Particle Insertion Location
    env_grid[x0,y0] = 2
    surroundings = [(x,y) for x in x0-1:x0+1 for y in y0-1:y0+1 if ((x!=2)|(y!=2))]
    surroundings = [CartesianIndex(x) for x in surroundings]

    particle_matrix = Array{Int64,2}(undef,n_agents,4)
    particle_matrix[:,1:2].=x0
    particle_matrix[:,3] = rand(0:359,n_agents) #heading; eg. angle
    particle_matrix[:,4] .= 0 #number of successful steps
    
    #Start Simulation
    for i in 1:n_iters
        #Sense Step
        theta_modifiers = (s_angle,0.0,-s_angle)
        sensors = Array{Int64,2}(undef,size(particle_matrix)[1],6)
        for j in 1:3
            xx = particle_matrix[:,1]
            yy = particle_matrix[:,2]
            thetas = particle_matrix[:,3].+theta_modifiers[j]

            sx = round.(Int,xx .+ s_offset.*cosd.(thetas))
            sy = round.(Int,yy .+ s_offset.*sind.(thetas))
            sxy = hcat(sx,sy)

            ix1 = 2*j-1
            ix2 = ix1+1
            sensors[:,ix1:ix2] = sxy
        end

        #Get Sensor Values and Update Headings
        sensor_values = Array{Float64,2}(undef,n_agents,3)
        for j in 1:n_agents
            temp = sensors[j,:]
            gvals = Array{Float64,1}(undef,3)
            for k in 1:3
                ix1 = 2*k-1
                ix2 = ix1+1
                sx,sy = temp[ix1],temp[ix2]
                try
                    sensor_val = trail_grid[sx,sy]
                    gvals[k] = sensor_val
                catch BoundsError
                    gvals[k] = 0.0
                end
            sensor_values[j,:] = gvals
            end
        end

        for j in 1:size(sensor_values)[1]
            rdraw = rand()
            lv,cv,rv = sensor_values[j,:]
            if rdraw<rt_chance
                particle_matrix[j,3] = rand(0:359)
            elseif lv>cv>=rv
                particle_matrix[j,3]+=s_angle
            elseif rv>cv>=lv
                particle_matrix[j,3]-=s_angle
            elseif lv==rv>cv
                particle_matrix[j,3] = rand(0:359)
            end
        end
        
        #Move Step
        idx = Random.randperm(n_agents)
        for j in idx
            temp = particle_matrix[j,:]
            px,py = temp[1],temp[2]
            theta = temp[3]

            x2 = round(Int,px+cosd(theta))
            y2 = round(Int,py+sind(theta))
            
            if ((x2<1)|(x2>grid_size))
                continue
            elseif ((y2<1)|(y2>grid_size))
                continue
            elseif x2==y2==x0
                continue
            elseif env_grid[x2,y2]==1
                continue
            else
                env_grid[x2,y2] = 1
                env_grid[px,py] = 0
                particle_matrix[j,1] = x2
                particle_matrix[j,2] = y2
                particle_matrix[j,4]+=1

                nearby = [(x,y) for x in x2-1:x2+1 for y in y2-1:y2+1]
                diffusion = Vector{Tuple}()
                for k in 1:length(nearby)
                    cc = nearby[k]
                    if cc[1]==cc[2]==x0
                        continue
                    elseif ((minimum(cc)<1)|(maximum(cc)>grid_size))
                        continue
                    else
                        push!(diffusion,cc)
                    end
                end

                #nearby = [CartesianIndex(x) for x in diffusion]
                trail_grid[x2,y2] = deposit_value
                #trail_grid[nearby] .= deposit_value
                density_grid[x2,y2] += deposit_value

            end
        end

        #Apply Kernel Filter
        trail_grid = kernel_filter(trail_grid)
        idx = findall(x->x>0,trail_grid)
        trail_grid[idx] .= trail_grid[idx].-signal_decomp

        idx = findall(x->x<0,trail_grid)
        trail_grid[idx] .= 0.0

    end

    return env_grid,trail_grid,density_grid
end


function PlannedSimulation(
    n_iters::Int64,
    grid_size::Int64,
    n_agents::Int64,
    n_food::Int64,
    food_size::Int64,
    s_angle::Int64,
    s_offset::Int64,
    rt_chance::Float64,
    signal_decomp::Float64,
    iter_reduce::Int64,
    reduction_value::Int64
    )
    #=
    :param n_iters: Number of iterations to run

    :param grid_size: Size of grid; grid is square

    :param n_agents: Number of particles to initialize

    :param: n_food: Number of food items to place on grid

    :param food_size: Side length of square

    :param s_angle: Angle of left and right sensors to particle location

    :param s_offset: Distance to sensor from particle location

    :param rt_chance: Chance of a random turn occurring

    :param signal_decomp: Value to subtract from trail map at each iteration.
    =#

    #Set Constants
    deposit_value = 1.10
    #drt = rt_chance/(n_agents÷8)

    if food_size%2==0
        food_size+=1
    end

    #Build Grids
    env_grid = zeros(Int64,grid_size,grid_size)
    trail_grid = zeros(Float64,grid_size,grid_size)
    density_grid = zeros(Float64,grid_size,grid_size) #Displays the sum of the trails

    x0=y0=grid_size÷2 #Particle Insertion Location
    env_grid[x0,y0] = 2
    surroundings = [(x,y) for x in x0-1:x0+1 for y in y0-1:y0+1 if ((x!=2)|(y!=2))]
    surroundings = [CartesianIndex(x) for x in surroundings]

    particle_matrix = Array{Int64,2}(undef,n_agents,3)
    particle_matrix[:,1:2].=x0
    particle_matrix[:,3] = rand(0:359,n_agents) #heading; eg. angle
    

    #Place Food on Environment and Trail Grids
    # food_coords = Array{Tuple{Int64,Int64},1}(undef,n_food)
    # let
    #     n1 = 0+grid_size÷4
    #     n2 = grid_size - grid_size÷4
    #     idx1 = [(x,y) for x in 1:n1 for y in 1:n1]
    #     idx2 = [(x,y) for x in n2:grid_size for y in n2:grid_size]
    #     idx = vcat(idx1,idx2)
    #     for i in 1:n_food
    #         food_coords[i] = StatsBase.sample(idx,1,replace=false)[1]
    #     end
    # end

    food_coords = [(170,170),(30,160),(165,25)]
    for i in 1:n_food
        nx,ny = food_coords[i]
        env_grid[nx,ny] = 2

        n = floor(Int,food_size/2)
        surroundings = [(x,y) for x in nx-n:nx+n for y in ny-n:ny+n]
        for j in 1:length(surroundings)
            try
                sx,sy = surroundings[j]
                trail_grid[sx,sy] = 2.0
            catch BoundsError
                continue
            end
        end
    end


    #Start Simulation
    for i in 1:n_iters
     #Sense Step
        theta_modifiers = (s_angle,0.0,-s_angle)
        sensors = Array{Int64,2}(undef,size(particle_matrix)[1],6)
        for j in 1:3
            xx = particle_matrix[:,1]
            yy = particle_matrix[:,2]
            thetas = particle_matrix[:,3].+theta_modifiers[j]

            sx = round.(Int,xx .+ s_offset.*cosd.(thetas))
            sy = round.(Int,yy .+ s_offset.*sind.(thetas))
            sxy = hcat(sx,sy)

            ix1 = 2*j-1
            ix2 = ix1+1
            sensors[:,ix1:ix2] = sxy
        end

        #Get Sensor Values and Update Headings
        sensor_values = Array{Float64,2}(undef,n_agents,3)
        for j in 1:n_agents
            temp = sensors[j,:]
            gvals = Array{Float64,1}(undef,3)
            for k in 1:3
                ix1 = 2*k-1
                ix2 = ix1+1
                sx,sy = temp[ix1],temp[ix2]
                try
                    sensor_val = trail_grid[sx,sy]
                    gvals[k] = sensor_val
                catch BoundsError
                    gvals[k] = 0.0
                end
            sensor_values[j,:] = gvals
            end
        end

        for j in 1:size(sensor_values)[1]
            rdraw = rand()
            lv,cv,rv = sensor_values[j,:]
            if rdraw<rt_chance
                particle_matrix[j,3] = rand(0:359)
            elseif lv>cv>=rv
                particle_matrix[j,3]+=s_angle
            elseif rv>cv>=lv
                particle_matrix[j,3]-=s_angle
            elseif lv==rv>cv
                particle_matrix[j,3] = rand(0:359)
            end
        end
        
        #Move Step
        idx = Random.randperm(n_agents)
        for j in idx
            temp = particle_matrix[j,:]
            px,py = temp[1],temp[2]
            theta = temp[3]

            x2 = round(Int,px+cosd(theta))
            y2 = round(Int,py+sind(theta))
            
            if ((x2<1)|(x2>grid_size))
                continue
            elseif ((y2<1)|(y2>grid_size))
                continue
            elseif x2==y2==x0
                continue
            elseif env_grid[x2,y2]==1
                continue
            else
                env_grid[x2,y2] = 1
                env_grid[px,py] = 0
                particle_matrix[j,1] = x2
                particle_matrix[j,2] = y2

                nearby = [(x,y) for x in x2-1:x2+1 for y in y2-1:y2+1]
                diffusion = Vector{Tuple{Int64,Int64}}()
                for k in 1:length(nearby)
                    cc = nearby[k]
                    if cc[1]==cc[2]==x0
                        continue
                    elseif ((minimum(cc)<1)|(maximum(cc)>grid_size))
                        continue
                    elseif trail_grid[cc[1],cc[2]]==2.0
                        continue
                    else
                        push!(diffusion,cc)
                    end
                end

                nearby = [CartesianIndex(x) for x in diffusion]
                if length(nearby)>0
                    trail_grid[nearby] .= 0.5
                end

                if trail_grid[x2,y2]!=2.0
                    trail_grid[x2,y2] = deposit_value
                end

                density_grid[x2,y2] += deposit_value

            end
        end
        trail_grid = kernel_filter(trail_grid)
        idx = findall(x->0.0<x<1.99,trail_grid)
        trail_grid[idx] .= trail_grid[idx].-signal_decomp

        idx = findall(x->x<0,trail_grid)
        trail_grid[idx] .= 0.0

        #rt_chance-=drt

        #Remove Particles that are Far from Food
        # if i%iter_reduce==0
        #     px,py = particle_matrix[:,1],particle_matrix[:,2]
        #     dist_list = Array{Float64,2}(undef,n_agents,n_food)
        #     for j in 1:n_food
        #         fx,fy = food_coords[j]
        #         dx = px.-fx
        #         dy = py.-fy
        #         dist = sqrt.(dx.^2 .+ dy.^2)
        #         dist_list[:,1] = dist
        #     end
            
        #     min_dists = vec(minimum(dist_list,dims=2))
        #     idx = sortperm(min_dists)
        #     idx2 = idx[1:end-reduction_value]
        #     idx3 = idx[(end-reduction_value)+1:end]
        #     temp = collect(zip(particle_matrix[idx3,1],particle_matrix[idx3,2]))
        #     temp = [CartesianIndex(x) for x in temp]
        #     env_grid[temp].=0

        #     #Update Particle Matrix
        #     pm1 = particle_matrix[idx2,:]
        #     pm2 = Array{Int64,2}(undef,reduction_value,3)
        #     pm2[:,1:2].=x0
        #     pm2[:,3] = rand(0:359,reduction_value)
        #     particle_matrix = vcat(pm1,pm2)
        # end         
    end

    return env_grid,trail_grid,density_grid    
end
    


#Update Particle Particle Matrix and Environment; Apoptosis
# idx = findall(x->x==n_steps,particle_matrix[:,4])
# if length(idx)>0
#     temp = particle_matrix[idx,1:2]
#     idx2 = collect(zip(temp[:,1],temp[:,2]))
#     grid_coords = [CartesianIndex(x) for x in idx2]
#     env_grid[grid_coords] .= 0

#     particle_matrix = particle_matrix[idx,:]
# end 

# if size(particle_matrix)[1]==0
#     break
#     print("All Particles are Gone!")
#     print(i)
# end    














            





        
    





    



    

        









    



